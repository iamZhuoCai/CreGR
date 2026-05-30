import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.dataset import AbstractDataset
# ===== [MOD] inherit from the patched MultiHeadVQVAETokenizer (tokenizer2)
# so _credibility_reg_loss already prefers `model.cred_head` when present =====
from genrec.tokenizers.MultiHeadVQVAE.tokenizer2 import (EmbDataset,
                                                          MultiHeadVQVAETokenizer)
from genrec.tokenizers.RQVAE.layers2 import RQVAE
# ===== [/MOD] =====


class RQVAETokenizer(MultiHeadVQVAETokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super().__init__(config, dataset)

    def load_sent_emb(self, dataset: AbstractDataset):
        """RQ-VAE embedding flow: concat(title+description, pretrain) without PCA."""
        if not hasattr(dataset, 'get_item_embeddings'):
            return super().load_sent_emb(dataset)

        self.log('[TOKENIZER][RQVAE] Using Disco title+description embeddings')
        sent_embs = dataset.get_item_embeddings().astype(np.float32)
        self.log('[TOKENIZER][RQVAE] Skipping PCA. Keep raw concatenated title+description embeddings.')

        pretrain_embs = None
        if hasattr(dataset, 'get_pretrain_item_embeddings'):
            pretrain_embs = dataset.get_pretrain_item_embeddings()
        if pretrain_embs is not None:
            pretrain_embs = pretrain_embs.astype(np.float32)
            if pretrain_embs.shape[0] != sent_embs.shape[0]:
                raise ValueError('Pretrain embeddings row count does not match item embeddings.')
            self.log('[TOKENIZER][RQVAE] Concatenating raw title+description embeddings with '
                     f'pretrain embeddings: {sent_embs.shape} + {pretrain_embs.shape}')
            sent_embs = np.concatenate([sent_embs, pretrain_embs], axis=1)
        else:
            self.log('[TOKENIZER][RQVAE] Pretrain embeddings unavailable; use title+description only.')

        return sent_embs

    def _cfg(self, rq_key: str, vq_key: str):
        if rq_key in self.config:
            return self.config[rq_key]
        return self.config[vq_key]

    def _train_rqvae_epoch(self, model: RQVAE, train_data,
                           optimizer: torch.optim.Optimizer,
                           known_labels=None,
                           current_epoch: int = 0):
        return self._train_vqvae_epoch(model,
                                       train_data,
                                       optimizer,
                                       known_labels=known_labels,
                                       current_epoch=current_epoch)

    def _train_rqvae(self,
                     sent_embs: torch.Tensor,
                     model_path: str,
                     raw_item_ids_for_training=None,
                     credible_items=None,
                     credreg_meta_path: str = None):
        device = self.config['device']
        use_cred_reg = bool(self.config.get('cred_reg_enable', False))
        use_cred_reg = (use_cred_reg and raw_item_ids_for_training is not None
                        and credible_items is not None)
        data = EmbDataset(sent_embs.cpu(), return_index=use_cred_reg)

        sk_eps = [0.0] * self.n_digit
        if self.n_digit > 0:
            sk_eps[-1] = float(self._cfg('rqvae_sk_epsilon',
                                         'vqvae_sk_epsilon'))

        # ============================================================
        # [MOD] Resolve cred-projection-head dimensions.
        #
        # Only build the head when cred regularization is actually on.
        # Defaults:
        #   cred_proj_dim    -> rqvae_e_dim (or vqvae_e_dim)
        #   cred_proj_hidden -> None  (head ctor falls back to e_dim)
        # ============================================================
        cred_proj_dim = None
        cred_proj_hidden = None
        if use_cred_reg:
            default_dim = self._cfg('rqvae_e_dim', 'vqvae_e_dim')
            cred_proj_dim = int(self.config.get('cred_proj_dim', default_dim))
            cred_proj_hidden_cfg = self.config.get('cred_proj_hidden', None)
            if cred_proj_hidden_cfg is not None:
                cred_proj_hidden = int(cred_proj_hidden_cfg)
            if cred_proj_dim <= 0:
                cred_proj_dim = 0
                cred_proj_hidden = None
                self.log('[TOKENIZER] cred_head disabled (cred_proj_dim<=0); '
                         'cred loss uses raw last residual (no MLP).')
        # ===== [/MOD] =====

        # ===== [MOD] cred residual-layer ablation =====
        # When a specific residual layer is requested for the credibility
        # loss there are two modes:
        #   * cred_reg_residual_through_head=False (default, legacy): disable
        #     the cred_head (cred_proj_dim=None) so _credibility_reg_loss takes
        #     its raw-residual branch (query_vecs = residual_k, no MLP).
        #   * cred_reg_residual_through_head=True: KEEP the cred_head; the loss
        #     then projects the layer-k residual through the MLP
        #     (query_vecs = cred_head(residual_k)). k=1 reduces to the legacy
        #     cred_head(x_e) baseline since residual_1 == x_e.
        cred_residual_layer = self.config.get('cred_reg_residual_layer', None)
        through_head = bool(
            self.config.get('cred_reg_residual_through_head', False))
        if use_cred_reg and cred_residual_layer not in (None, 0, '', 'None'):
            if through_head:
                self.log('[TOKENIZER] cred_reg_residual_layer='
                         f'{cred_residual_layer}: routing layer-k residual '
                         'THROUGH cred_head MLP (cred_head kept).')
            else:
                cred_proj_dim = None
                cred_proj_hidden = None
                self.log('[TOKENIZER] cred_reg_residual_layer='
                         f'{cred_residual_layer}: using residual-path '
                         'credibility loss (cred_head disabled).')
        # ===== [/MOD] =====

        rqvae_model = RQVAE(
            in_dim=data.dim,
            num_emb_list=self.codebook_sizes,
            e_dim=self._cfg('rqvae_e_dim', 'vqvae_e_dim'),
            layers=self._cfg('rqvae_hidden_sizes', 'vqvae_hidden_sizes'),
            title_desc_dim=int(self.config.get('rqvae_title_desc_dim', 8192)),
            pretrain_dim=int(self.config.get('rqvae_pretrain_dim', 64)),
            td_reduce_dim=int(self.config.get('rqvae_title_desc_reduce_dim', 64)),
            pre_reduce_dim=int(self.config.get('rqvae_pretrain_reduce_dim', 64)),
            dropout_prob=self._cfg('rqvae_dropout', 'vqvae_dropout'),
            bn=self._cfg('rqvae_bn', 'vqvae_bn'),
            loss_type='mse',
            quant_loss_weight=self._cfg('rqvae_quant_weight',
                                        'vqvae_quant_weight'),
            beta=self._cfg('rqvae_beta', 'vqvae_beta'),
            kmeans_init=self.config['kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            sk_epsilons=sk_eps,
            sk_iters=self._cfg('rqvae_sk_iters', 'vqvae_sk_iters'),
            # ===== [MOD] wire the new head into model construction =====
            cred_proj_dim=cred_proj_dim,
            cred_proj_hidden=cred_proj_hidden,
            # ===== [/MOD] =====
        ).to(device)
        self.log(rqvae_model)

        batch_size = self._cfg('rqvae_batch_size', 'vqvae_batch_size')
        num_epochs = self._cfg('rqvae_epoch', 'vqvae_epoch')
        verbose = self._cfg('rqvae_verbose', 'vqvae_verbose')

        rqvae_l2 = float(self._cfg('rqvae_l2', 'vqvae_l2'))
        optimizer_name = str(
            self._cfg('rqvae_optimizer', 'vqvae_optimizer')).lower()
        rqvae_lr = self._cfg('rqvae_lr', 'vqvae_lr')
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(rqvae_model.parameters(),
                                         lr=rqvae_lr,
                                         weight_decay=rqvae_l2)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(rqvae_model.parameters(),
                                          lr=rqvae_lr,
                                          weight_decay=rqvae_l2)
        else:
            raise ValueError("Optimizer not supported")

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
                     f'hidden={cred_proj_hidden} (None=e_dim)')
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
        best_intra_cred = float('inf')
        best_collision_epoch = 0
        start_epoch = 0

        resume_rqvae_ckpt = self.config.get('resume_rqvae_ckpt', None)
        if resume_rqvae_ckpt:
            self.log(f"[TOKENIZER] Resuming RQ-VAE from checkpoint: {resume_rqvae_ckpt}")
            checkpoint = torch.load(resume_rqvae_ckpt,
                                    map_location=device,
                                    weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # ===== [MOD] tolerate cred_head weight mismatch on resume =====
                # Old checkpoints (no cred_head) cannot strict-load into the
                # patched model, and new checkpoints with a different
                # proj/hidden config also can't strict-load. We therefore
                # load with strict=False and let the cred_head re-initialize
                # if it doesn't match. Non-cred params still load normally.
                missing, unexpected = rqvae_model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False)
                if missing or unexpected:
                    self.log(f"[TOKENIZER] Resume strict=False — missing keys: "
                             f"{missing}; unexpected keys: {unexpected}")
                # ===== [/MOD] =====
                if 'optimizer_state_dict' in checkpoint:
                    # ===== [MOD] optimizer state may not match new cred_head
                    # params; wrap in try/except so resume still works =====
                    try:
                        optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict'])
                    except (ValueError, KeyError) as e:
                        self.log(f"[TOKENIZER] Skipped optimizer state restore: {e}")
                    # ===== [/MOD] =====
                start_epoch = int(checkpoint.get('epoch', 0))
                lr_resume = float(rqvae_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_resume
                best_loss = float(checkpoint.get('best_loss', best_loss))
                best_collision_rate = float(
                    checkpoint.get('best_collision_rate', best_collision_rate))
                best_intra_cred = float(
                    checkpoint.get('best_intra_cred', best_intra_cred))
                best_collision_epoch = int(
                    checkpoint.get('best_collision_epoch', best_collision_epoch))
            else:
                # Compatibility fallback: allow loading plain model state_dict
                # ===== [MOD] same strict=False tolerance for plain state_dicts =====
                rqvae_model.load_state_dict(checkpoint, strict=False)
                # ===== [/MOD] =====
            self.log(f"[TOKENIZER] Resume start epoch: {start_epoch}")

        self.log("[TOKENIZER] Training RQ-VAE model...")
        rqvae_model.train()

        rq_train_wall_start = time.perf_counter()
        for epoch in tqdm(range(start_epoch, num_epochs)):
            epoch_wall_start = time.perf_counter()
            train_wall_start = time.perf_counter()
            (train_loss, train_recon_loss, train_quant_loss,
             train_cred_reg_loss, train_intra_cred, train_intra_uncred,
             train_inter, cred_reg_steps,
             total_count) = self._train_rqvae_epoch(
                 rqvae_model,
                 dataloader,
                 optimizer,
                 known_labels=known_labels,
                 current_epoch=epoch)
            train_wall_s = time.perf_counter() - train_wall_start
            collision_wall_s = 0.0

            if (epoch + 1) % verbose == 0:
                collision_wall_start = time.perf_counter()
                collision_rate = self._valid_collision(rqvae_model, dataloader)
                collision_wall_s = time.perf_counter() - collision_wall_start

                save_method = self._cfg('rqvae_save_method', 'vqvae_save_method')
                cred_reg_start_epoch = int(
                    self.config.get('cred_reg_start_epoch', 0))
                intra_cred_metric = train_intra_cred / max(1, cred_reg_steps)
                # For MHMisinfo we prefer selecting checkpoint by minimal intra-cred.
                if (self.config.get('category') == 'MHMisinfo'
                        and save_method == 'collision'):
                    save_method = 'intra_cred'

                if save_method == 'loss':
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_intra_cred = intra_cred_metric
                        best_collision_epoch = epoch + 1
                        torch.save(rqvae_model.state_dict(), model_path)
                elif save_method == 'collision':
                    if collision_rate < best_collision_rate:
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_intra_cred = intra_cred_metric
                        best_collision_epoch = epoch + 1
                        torch.save(rqvae_model.state_dict(), model_path)
                elif save_method == 'intra_cred':
                    # Only select by intra-cred after credibility loss is enabled.
                    if ((epoch + 1) >= cred_reg_start_epoch
                            and cred_reg_steps > 0
                            and intra_cred_metric < best_intra_cred):
                        best_loss = train_loss
                        best_collision_rate = collision_rate
                        best_intra_cred = intra_cred_metric
                        best_collision_epoch = epoch + 1
                        torch.save(rqvae_model.state_dict(), model_path)
                elif save_method == 'last':
                    best_loss = train_loss
                    best_collision_rate = collision_rate
                    best_intra_cred = intra_cred_metric
                    best_collision_epoch = epoch + 1
                    torch.save(rqvae_model.state_dict(), model_path)
                else:
                    raise ValueError("Save method not supported")

                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Timing (s): train_step={train_wall_s:.4f}, collision_eval={collision_wall_s:.4f}, epoch_total={(time.perf_counter() - epoch_wall_start):.4f}\n"
                    f"\t  Training loss: {train_loss}\n"
                    f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {train_recon_loss}\n"
                    f"\t  Quantization loss: {train_quant_loss}\n"
                    f"\t  Credibility reg loss: {train_cred_reg_loss}\n"
                    f"\t  Intra cred: {train_intra_cred / max(1, cred_reg_steps)}\n"
                    f"\t  Intra uncred: {train_intra_uncred / max(1, cred_reg_steps)}\n"
                    f"\t  Inter: {train_inter / max(1, cred_reg_steps)}\n"
                    f"\t  Collision rate: {collision_rate}\n")

            # Always keep a resumable checkpoint at cred_reg_start_epoch.
            cred_reg_start_epoch = int(self.config.get('cred_reg_start_epoch', 0))
            if (cred_reg_start_epoch > 0
                    and (epoch + 1) == cred_reg_start_epoch):
                checkpoint_path = (
                    f"{os.path.splitext(model_path)[0]}.ep{cred_reg_start_epoch}.ckpt.pth"
                )
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': rqvae_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'best_collision_rate': best_collision_rate,
                    'best_intra_cred': best_intra_cred,
                    'best_collision_epoch': best_collision_epoch,
                    'config': self.config,
                }
                torch.save(checkpoint, checkpoint_path)
                self.log(
                    f"[TOKENIZER] Saved epoch-{cred_reg_start_epoch} checkpoint to "
                    f"{checkpoint_path}")

        rq_train_elapsed_s = time.perf_counter() - rq_train_wall_start
        self.log(
            "[TOKENIZER] RQ-VAE wall time from training loop start "
            f"(epoch {start_epoch + 1}) through epoch {num_epochs}: "
            f"{rq_train_elapsed_s:.3f} s ({rq_train_elapsed_s / 60.0:.3f} min, "
            f"{rq_train_elapsed_s / 3600.0:.4f} h)"
        )

        self.log("[TOKENIZER] RQ-VAE training complete.")
        self.log(
            f"Best Epoch: {best_collision_epoch} Best Collision Rate: {best_collision_rate} "
            f"Best Intra Cred: {best_intra_cred} Best Loss: {best_loss}"
        )

        # Always save a resumable checkpoint at the final epoch so training can continue later.
        final_epoch = int(num_epochs)
        final_checkpoint_path = (
            f"{os.path.splitext(model_path)[0]}.ep{final_epoch}.ckpt.pth")
        final_checkpoint = {
            'epoch': final_epoch,
            'model_state_dict': rqvae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_collision_rate': best_collision_rate,
            'best_intra_cred': best_intra_cred,
            'best_collision_epoch': best_collision_epoch,
            'config': self.config,
        }
        torch.save(final_checkpoint, final_checkpoint_path)
        self.log(f"[TOKENIZER] Saved final epoch checkpoint to {final_checkpoint_path}")

        # `model_path` is updated during training by best loss / collision / intra_cred
        # selection. For semantic ID export we use the **last training epoch** weights
        # (in-memory model), then persist them to `model_path`.
        torch.save(rqvae_model.state_dict(), model_path)
        self.log(
            f'[TOKENIZER] Wrote final-epoch (ep={final_epoch}) weights to {model_path} '
            'for semantic ID generation (not best intra_cred / collision snapshot).')
        return rqvae_model

    # Compatibility hook: parent pipeline calls _train_vqvae.
    def _train_vqvae(self,
                     sent_embs: torch.Tensor,
                     model_path: str,
                     raw_item_ids_for_training=None,
                     credible_items=None,
                     credreg_meta_path: str = None):
        return self._train_rqvae(sent_embs, model_path,
                                 raw_item_ids_for_training, credible_items,
                                 credreg_meta_path)
