import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.tokenizer import SemIDTokenizer
# ===== [MOD] use the patched MultiHeadVQVAE that exposes cred_head =====
from genrec.tokenizers.MultiHeadVQVAE.layers2 import MultiHeadVQVAE
# ===== [/MOD] =====


class EmbDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings: torch.Tensor, return_index: bool = False):
        self.embeddings = embeddings
        self.dim = self.embeddings.shape[-1]
        self.return_index = return_index

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        if self.return_index:
            return tensor_emb, int(index)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class MultiHeadVQVAETokenizer(SemIDTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(MultiHeadVQVAETokenizer, self).__init__(config, dataset)

        self.id2item = dataset.id_mapping['id2item']

        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = int(np.sum(self.codebook_sizes) + 1)
        self.ignored_label = -100

    @property
    def n_digit(self):
        return self.config['vq_n_codebooks']

    @property
    def codebook_sizes(self):
        if isinstance(self.config['vq_codebook_size'], list):
            return self.config['vq_codebook_size']
        else:
            return [self.config['vq_codebook_size']] * self.n_digit

    @torch.no_grad()
    def _valid_collision(self, vqvae_model, dataloader):
        vqvae_model.eval()

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(dataloader):
            if isinstance(data, (list, tuple)):
                data = data[0]
            num_sample += len(data)
            data = data.to(self.config['device'])
            indices = vqvae_model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set))) / num_sample

        return collision_rate

    def _build_known_credibility_labels(self, raw_item_ids_for_training,
                                        credible_items):
        labels = np.full(len(raw_item_ids_for_training), -1, dtype=np.int64)
        if len(raw_item_ids_for_training) == 0:
            return labels, [], []

        credible_set = set(int(x) for x in np.asarray(credible_items).tolist())
        cred_idx = []
        uncred_idx = []
        for i, rid in enumerate(raw_item_ids_for_training):
            if int(rid) in credible_set:
                cred_idx.append(i)
            else:
                uncred_idx.append(i)

        known_ratio = float(self.config.get('cred_known_ratio', 0.2))
        seed = int(self.config.get('cred_reg_seed', 2024))
        rng = np.random.default_rng(seed)

        n_cred = max(1, int(len(cred_idx) * known_ratio)) if cred_idx else 0
        n_uncred = max(1,
                       int(len(uncred_idx) * known_ratio)) if uncred_idx else 0

        if n_cred > 0:
            for idx in rng.choice(cred_idx, size=n_cred, replace=False):
                labels[int(idx)] = 1
        if n_uncred > 0:
            for idx in rng.choice(uncred_idx, size=n_uncred, replace=False):
                labels[int(idx)] = 0

        known_uncredible_raw_ids = []
        if len(uncred_idx) > 0:
            known_uncredible_raw_ids = [
                int(raw_item_ids_for_training[i])
                for i in np.where(labels == 0)[0].tolist()
            ]
        known_credible_raw_ids = []
        if len(cred_idx) > 0:
            known_credible_raw_ids = [
                int(raw_item_ids_for_training[i])
                for i in np.where(labels == 1)[0].tolist()
            ]

        self.log('[TOKENIZER] Credibility-aware reg labels: '
                 f'known credible={int((labels == 1).sum())}, '
                 f'known uncredible={int((labels == 0).sum())}, '
                 f'unknown={int((labels < 0).sum())}')
        return labels, known_uncredible_raw_ids, known_credible_raw_ids

    def _credibility_reg_loss(self, model: MultiHeadVQVAE, indices,
                              batch_labels, batch_inputs):
        if indices.ndim != 2 or indices.shape[1] < 1:
            z = torch.zeros((), device=indices.device)
            return z, z, z, z

        known = batch_labels >= 0
        if int(known.sum().item()) < 2:
            z = torch.zeros((), device=indices.device)
            return z, z, z, z

        # Build the query vector for the last quantizer.
        # For RQ-VAE this is the last residual vector (before the final
        # residual quantizer). For MultiHeadVQVAE this falls back to the last
        # head input chunk from encoder output.
        if (not hasattr(model, 'vq') or not hasattr(model.vq, 'vq_layers')
                or len(model.vq.vq_layers) == 0):
            raise ValueError('Model does not expose vq.vq_layers for '
                             'query-vector lookup.')
        if not hasattr(model, 'encoder'):
            raise ValueError('Model does not expose encoder for '
                             'query-vector lookup.')

        if hasattr(model, 'encode_inputs'):
            x_e = model.encode_inputs(batch_inputs)
        else:
            x_e = model.encoder(batch_inputs)
        query_vecs = None

        # RQ-VAE path: last residual (before final quantizer), optionally
        # projected through cred_head so cred grads do not touch the codebook.
        if getattr(model, 'cred_head', None) is not None:
            query_vecs = model.cred_head(x_e)
        elif x_e.ndim == 2 and x_e.shape[-1] == model.vq.vq_layers[0].e_dim:
            residual = x_e
            for q_idx, quantizer in enumerate(model.vq.vq_layers[:-1]):
                q_ids = indices[:, q_idx]
                q_emb = quantizer.embedding(q_ids)
                residual = residual - q_emb.detach()
            query_vecs = residual
        else:
            # MultiHeadVQVAE fallback: use last head chunk as query vector.
            num_q = len(model.vq.vq_layers)
            if x_e.ndim != 2 or x_e.shape[-1] % num_q != 0:
                raise ValueError('Unexpected encoder output shape for '
                                 'query-vector lookup.')
            head_chunks = torch.chunk(x_e, num_q, dim=1)
            query_vecs = head_chunks[-1]

        # Normalize residual query vectors before distance computation to
        # stabilize scale and avoid exploding center-distance terms.
        query_vecs = F.normalize(query_vecs, p=2, dim=-1, eps=1e-8)

        cred_mask = batch_labels == 1
        uncred_mask = batch_labels == 0
        if int(cred_mask.sum().item()) == 0 or int(uncred_mask.sum().item()) == 0:
            z = torch.zeros((), device=indices.device)
            return z, z, z, z

        cred_embs = query_vecs[cred_mask]
        uncred_embs = query_vecs[uncred_mask]

        # Contrastive-style MSE objective:
        # 1) Pull each class toward its own center.
        # 2) Push class centers apart with a hinge margin.
        center_cred = cred_embs.mean(dim=0, keepdim=True)
        center_uncred = uncred_embs.mean(dim=0, keepdim=True)

        intra_cred = F.mse_loss(cred_embs, center_cred.expand_as(cred_embs))
        intra_uncred = F.mse_loss(uncred_embs,
                                  center_uncred.expand_as(uncred_embs))

        # margin = float(self.config.get('cred_reg_margin', 1.0))
        # center_dist = F.mse_loss(center_cred, center_uncred)
        # inter = F.relu(torch.tensor(margin, device=indices.device) - center_dist)
        inter = - F.mse_loss(center_cred, center_uncred)
        inter_w = float(self.config.get('cred_reg_inter_weight', 100))
        total = intra_cred + intra_uncred + inter_w * inter
        return total, intra_cred, intra_uncred, inter

        # def pairwise_intra_mse(x):
        #     # Pull every vector close to all other same-class vectors.
        #     if x.shape[0] < 2:
        #         return torch.zeros((), device=x.device)
        #     diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
        #     sq_dist = diffs.pow(2).mean(dim=-1)  # per-pair MSE
        #     mask = ~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        #     return sq_dist[mask].mean()

        # intra = pairwise_intra_mse(cred_embs) + pairwise_intra_mse(uncred_embs)

        # # Push every credible vector away from every uncredible vector.
        # # Inverse-distance objective: minimizing this term encourages
        # # pairwise MSE to become larger.
        # pairwise_mse = ((cred_embs[:, None, :] - uncred_embs[None, :, :])**2
        #                 ).mean(dim=-1)
        # eps = 1e-8
        # inter = 1e-6 * (1.0 / (pairwise_mse + eps)).mean()
        # return intra + inter

    def _train_vqvae_epoch(self, model: MultiHeadVQVAE, train_data,
                           optimizer: torch.optim.Optimizer,
                           known_labels: np.ndarray = None,
                           current_epoch: int = 0):
        model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        total_count = 0

        total_cred_reg_loss = 0
        total_intra_cred = 0
        total_intra_uncred = 0
        total_inter = 0
        cred_reg_steps = 0

        for batch_idx, data in enumerate(train_data):
            batch_item_idx = None
            if isinstance(data, (list, tuple)):
                data, batch_item_idx = data
            data = data.to(self.config['device'])
            optimizer.zero_grad()
            out, vq_loss, indices, unused_cnt = model(data)
            loss, loss_recon, quant_loss = model.compute_loss(out,
                                                              vq_loss,
                                                              xs=data)

            cred_reg_loss = torch.zeros((), device=data.device)
            intra_cred = torch.zeros((), device=data.device)
            intra_uncred = torch.zeros((), device=data.device)
            inter = torch.zeros((), device=data.device)
            cred_reg_start_epoch = int(self.config.get('cred_reg_start_epoch', 0))
            enable_cred_reg = ((current_epoch + 1) >= cred_reg_start_epoch)
            if (enable_cred_reg and known_labels is not None
                    and batch_item_idx is not None):
                batch_item_idx_np = np.asarray(batch_item_idx)
                batch_labels = torch.from_numpy(
                    known_labels[batch_item_idx_np]).to(data.device)
                cred_reg_loss, intra_cred, intra_uncred, inter = self._credibility_reg_loss(
                    model, indices, batch_labels, data)
                loss = loss + float(self.config.get('cred_reg_weight', 0.1)
                                    ) * cred_reg_loss

            if torch.isnan(loss):
                raise ValueError("Training loss is nan")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += quant_loss.item()
            total_cred_reg_loss += cred_reg_loss.item()
            total_intra_cred += intra_cred.item()
            total_intra_uncred += intra_uncred.item()
            total_inter += inter.item()
            if batch_item_idx is not None:
                cred_reg_steps += 1
            total_count += unused_cnt

        return (total_loss, total_recon_loss, total_quant_loss,
                total_cred_reg_loss, total_intra_cred, total_intra_uncred,
                total_inter, cred_reg_steps, total_count)

    def _train_vqvae(self,
                     sent_embs: torch.Tensor,
                     model_path: str,
                     raw_item_ids_for_training=None,
                     credible_items=None,
                     credreg_meta_path: str = None):
        device = self.config['device']
        use_cred_reg = bool(self.config.get('cred_reg_enable', False))
        use_cred_reg = use_cred_reg and raw_item_ids_for_training is not None and credible_items is not None
        data = EmbDataset(sent_embs.cpu(), return_index=use_cred_reg)

        # ============================================================
        # [MOD] Resolve cred-projection-head dimensions.
        #
        # Only build the head when cred regularization is actually on, so
        # we don't waste params during plain VQ-VAE training. Defaults:
        #   cred_proj_dim    -> vqvae_e_dim   (same scale as a single head)
        #   cred_proj_hidden -> None          (head ctor falls back to
        #                                      encoder_out_dim)
        # ============================================================
        cred_proj_dim = None
        cred_proj_hidden = None
        if use_cred_reg:
            cred_proj_dim = int(self.config.get('cred_proj_dim',
                                                self.config['vqvae_e_dim']))
            cred_proj_hidden_cfg = self.config.get('cred_proj_hidden', None)
            if cred_proj_hidden_cfg is not None:
                cred_proj_hidden = int(cred_proj_hidden_cfg)
        # ===== [/MOD] =====

        vqvae_model = MultiHeadVQVAE(
            in_dim=data.dim,
            num_emb_list=self.codebook_sizes,
            e_dim=self.config['vqvae_e_dim'],
            layers=self.config['vqvae_hidden_sizes'],
            dropout_prob=self.config['vqvae_dropout'],
            bn=self.config['vqvae_bn'],
            loss_type='mse',
            quant_loss_weight=self.config['vqvae_quant_weight'],
            beta=self.config['vqvae_beta'],
            kmeans_init=self.config['kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            sk_epsilons=[0, 0, 0, self.config['vqvae_sk_epsilon']],
            sk_iters=self.config['vqvae_sk_iters'],
            # ===== [MOD] wire the new head into model construction =====
            cred_proj_dim=cred_proj_dim,
            cred_proj_hidden=cred_proj_hidden,
            # ===== [/MOD] =====
        ).to(device)
        self.log(vqvae_model)

        # Model training
        batch_size = self.config['vqvae_batch_size']
        num_epochs = self.config['vqvae_epoch']
        verbose = self.config['vqvae_verbose']


        vqvae_l2 = float(self.config['vqvae_l2'])
        if self.config['vqvae_optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(vqvae_model.parameters(),
                                         lr=self.config['vqvae_lr'],
                                         weight_decay=vqvae_l2)
        elif self.config['vqvae_optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(vqvae_model.parameters(),
                                          lr=self.config['vqvae_lr'],
                                          weight_decay=vqvae_l2)
        else:
            raise ValueError("Optimizer not supported")
        # optimizer = torch.optim.Adagrad(vqvae_model.parameters(),
        #                                 lr=self.config['vqvae_lr'])
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)

        known_labels = None
        known_uncredible_raw_ids = []
        known_credible_raw_ids = []
        if use_cred_reg:
            known_labels, known_uncredible_raw_ids, known_credible_raw_ids = self._build_known_credibility_labels(
                raw_item_ids_for_training, credible_items)
            self.log('[TOKENIZER] Credibility-aware regularization enabled. '
                     f'weight={self.config.get("cred_reg_weight", 0.1)}, '
                     f'margin={self.config.get("cred_reg_margin", 1.0)}, '
                     f'known_ratio={self.config.get("cred_known_ratio", 0.2)}')
            # ===== [MOD] log cred head dims so runs are auditable =====
            self.log('[TOKENIZER] Credibility projection head: '
                     f'proj_dim={cred_proj_dim}, '
                     f'hidden={cred_proj_hidden} (None=encoder_out_dim)')
            # ===== [/MOD] =====
            if credreg_meta_path is not None:
                meta = {
                    'known_uncredible_raw_ids': known_uncredible_raw_ids,
                    'known_credible_raw_ids': known_credible_raw_ids,
                    'cred_known_ratio': float(self.config.get(
                        'cred_known_ratio', 0.2)),
                    'cred_reg_seed': int(self.config.get('cred_reg_seed',
                                                         2024)),
                }
                with open(credreg_meta_path, 'w') as f:
                    json.dump(meta, f)
                self.log('[TOKENIZER] Saved credibility reg metadata to '
                         f'{credreg_meta_path}')

        best_loss = float('inf')
        best_collision_rate = float('inf')
        best_collision_epoch = 0

        self.log("[TOKENIZER] Training MultiVQ-VAE model...")

        vqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            (train_loss, train_recon_loss, train_quant_loss,
             train_cred_reg_loss, train_intra_cred, train_intra_uncred,
             train_inter, cred_reg_steps,
             total_count) = self._train_vqvae_epoch(
                 vqvae_model,
                 dataloader,
                 optimizer,
                 known_labels=known_labels,
                 current_epoch=epoch)

            if (epoch + 1) % verbose == 0:
                collision_rate = self._valid_collision(vqvae_model, dataloader)

                if self.config['vqvae_save_method'] == 'loss':
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_collision_epoch = epoch + 1
                        torch.save(vqvae_model.state_dict(), model_path)
                elif self.config['vqvae_save_method'] == 'collision':
                    if collision_rate < best_collision_rate:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_collision_epoch = epoch + 1
                        torch.save(vqvae_model.state_dict(), model_path)
                elif self.config['vqvae_save_method'] == 'last':
                    best_loss = train_loss
                    best_collision_rate = collision_rate
                    best_collision_epoch = epoch + 1
                    torch.save(vqvae_model.state_dict(), model_path)
                else:
                    raise ValueError("Save method not supported")

                self.log(
                    f"[TOKENIZER] VQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {train_loss}\n"
                    f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {train_recon_loss}\n"
                    f"\t  Quantization loss: {train_quant_loss}\n"
                    f"\t  Credibility reg loss: {train_cred_reg_loss}\n"
                    f"\t  Intra cred: {train_intra_cred / max(1, cred_reg_steps)}\n"
                    f"\t  Intra uncred: {train_intra_uncred / max(1, cred_reg_steps)}\n"
                    f"\t  Inter: {train_inter / max(1, cred_reg_steps)}\n"
                    f"\t  Collision rate: {collision_rate}\n")

        self.log("[TOKENIZER] VQ-VAE training complete.")

        self.log(
            f"Best Epoch: {best_collision_epoch} Best Collision Rate: {best_collision_rate} Best Loss: {best_loss}"
        )

        vqvae_model.load_state_dict(torch.load(model_path))
        return vqvae_model

    def _check_collision(self, str_sem_ids):
        tot_item = len(str_sem_ids)
        tot_ids = len(set(str_sem_ids.tolist()))
        self.log(
            f'[TOKENIZER] Collision rate: {(tot_item - tot_ids) / tot_item}')
        return tot_item == tot_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        if self.config['sem_ids_path'] is None:
            sem_ids_path = os.path.join(self.config['result_dir'],
                                        self.config['sem_ids_dir'],
                                        f"{self.config['run_time']}.sem_ids")
            os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        else:
            token_model_dir = (
                f"TokenModel_Seq-{self.config['max_item_seq_len']}_"
                f"{self.config['tokenizer_name']}")
            sem_ids_path = os.path.join(
                self.config['base_result_dir'],
                f"{self.config['dataset']}_{self.config['category']}",
                token_model_dir,
                self.config['sem_ids_dir'], self.config['sem_ids_path'])

        if not os.path.exists(sem_ids_path):
            self.log(f"{sem_ids_path} not found. Generating semantic IDs...")

            sent_embs = self.load_sent_emb(dataset)
            self.log(
                f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)

            self.log(
                f'[TOKENIZER] Semantic IDs not found. Training VQ-VAE model...'
            )
            embs_for_training = torch.FloatTensor(
                sent_embs[training_item_mask]).to(self.config['device'])
            sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])

            training_item_indices = np.where(training_item_mask)[0]
            raw_item_ids_for_training = []
            for idx in training_item_indices:
                item = self.id2item[int(idx) + 1]
                raw_item_ids_for_training.append(int(item))

            model_path = os.path.join(self.config['result_dir'],
                                      self.config['ckpt_dir'],
                                      f"{self.config['run_time']}.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            credreg_meta_path = f'{sem_ids_path}.credreg_meta.json'
            credible_items = getattr(dataset, 'credible_items', None)
            vqvae_model = self._train_vqvae(
                embs_for_training,
                model_path,
                raw_item_ids_for_training=raw_item_ids_for_training,
                credible_items=credible_items,
                credreg_meta_path=credreg_meta_path)
            self._generate_semantic_id(vqvae_model, sent_embs, sem_ids_path)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _generate_semantic_id(self, vqvae_model: MultiHeadVQVAE,
                              sent_embs: torch.Tensor,
                              sem_ids_path: str) -> None:
        vqvae_model.eval()

        # Compatibility fallback: RQVAE configs use rqvae_* keys.
        batch_size = self.config.get('rqvae_batch_size',
                                     self.config.get('vqvae_batch_size', 2048))
        data = EmbDataset(sent_embs.cpu())
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True)

        all_indices = []
        all_indices_str = []

        for d in tqdm(dataloader):
            d = d.to(self.config['device'])

            indices = vqvae_model.get_indices(d, use_sk=False)

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = index.tolist()

                all_indices.append(code)
                all_indices_str.append('-'.join([str(x) for x in code]))

        all_indices = np.array(all_indices)
        all_indices_str = np.array(all_indices_str)

        sk_epsilon = self.config.get('rqvae_sk_epsilon',
                                     self.config.get('vqvae_sk_epsilon', 0.0))
        sk_iters = self.config.get('rqvae_sk_iters',
                                   self.config.get('vqvae_sk_iters', 0))
        use_sk = (sk_epsilon > 0) and (sk_iters > 0)

        if use_sk:
            for _ in range(30):
                # for _ in range(1):
                if self._check_collision(all_indices_str):
                    break

                collision_item_groups = self._get_collision_items(
                    all_indices_str)

                for collision_items in collision_item_groups:
                    d = data[collision_items].to(self.config['device'])
                    indices = vqvae_model.get_indices(d, use_sk=True)

                    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                    for item, index in zip(collision_items, indices):
                        code = index.tolist()

                        all_indices[item] = code
                        all_indices_str[item] = '-'.join(
                            [str(x) for x in code])

        self.log(f"All indices number: {len(all_indices)}")

        indices_count = defaultdict(int)
        for index in all_indices_str:
            indices_count[index] += 1

        # self.log("Max number of conflicts: ", max(indices_count.values()))
        self.log(f"Max number of conflicts: {max(indices_count.values())}")

        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        self.log("Collision Rate :{}".format(
            (tot_item - tot_indice) / tot_item))

        item2sem_ids = {}
        for item_id, indices in enumerate(all_indices.tolist()):
            item = self.id2item[int(item_id) + 1]
            item2sem_ids[item] = list(indices)

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            # json.dump(all_indices_str, f)
            json.dump(item2sem_ids, f)

    def _get_collision_items(self, str_sem_ids):
        sem_id2item = defaultdict(list)
        for i, str_sem_id in enumerate(str_sem_ids):
            sem_id2item[str_sem_id].append(i)

        collision_item_groups = []
        for str_sem_id in sem_id2item:
            if len(sem_id2item[str_sem_id]) > 1:
                collision_item_groups.append(sem_id2item[str_sem_id])

        return collision_item_groups

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        if item2sem_ids:
            first_sem_ids = next(iter(item2sem_ids.values()))
            actual_n_digit = len(first_sem_ids)
            if actual_n_digit != self.n_digit:
                self.log('[TOKENIZER] Loaded semantic IDs use '
                         f'{actual_n_digit} codebooks, overriding config '
                         f'vq_n_codebooks={self.n_digit}.')
                self.config['vq_n_codebooks'] = actual_n_digit
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] +
                                  self.codebook_sizes[digit - 1])
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids
