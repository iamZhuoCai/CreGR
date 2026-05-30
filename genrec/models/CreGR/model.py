from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import log
from CreGR import CreGRConfig, CreGRModelLM
from CreGR.modeling_cregr import (CausalLMOutputWithPast, ModuleType,
                                  init_weights)


@dataclass
class CreGROutPut:
    loss: torch.Tensor
    his_mask_loss: torch.Tensor
    target_mask_loss: torch.Tensor


class CreGR(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(CreGR, self).__init__(config, dataset, tokenizer)

        self.logger = getLogger()

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        self.posValidTokens = self.get_posValidToken().to(
            self.config['device'])

        self.mask_token_id = tokenizer.mask_token
        cregrconfig = CreGRConfig(
            activation_type=config['activation_type'],
            attention_dropout=config['dropout_rate'],
            bias_for_layer_norm=config['bias_for_layer_norm'],
            block_type=config['block_type'],
            d_model=config['d_model'],
            mlp_ratio=config['mlp_ratio'],
            embedding_dropout=config['dropout_rate'],
            init_fn=config['init_fn'],
            layer_norm_type=config['layer_norm_type'],
            max_sequence_length=(config['max_item_seq_len'] + 1) *
            self.tokenizer.n_digit,
            n_heads=config['n_heads'],
            n_kv_heads=config['n_kv_heads'],
            n_layers=config['n_layers'],
            residual_dropout=config['dropout_rate'],
            rope=config['rope'],
            rope_theta=config['rope_theta'],
            weight_tying=config['weight_tying'],
            vocab_size=tokenizer.vocab_size,
            embedding_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            mask_token_id=self.mask_token_id)

        self.cregr = CreGRModelLM(cregrconfig)

        self.item_pos_emb = nn.Embedding(
            num_embeddings=config['max_item_seq_len'] + 1,
            embedding_dim=config['d_model'])
        init_weights(self.cregr.model.config,
                     self.item_pos_emb,
                     type_of_module=ModuleType.emb)

        self.temperature = self.config['temperature']
        self.log("scale logits by temperature: {}".format(self.temperature))

        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label, reduction='none')
        self.his_mask_w = self.config['his_mask_w']

        self.gen_steps = config['gen_steps']

        self.val_num_beams = config['val_num_beams']
        self.log(f"using num beams: {self.val_num_beams} for validation")
        self.num_beams = config['num_beams']
        self._init_credibility_item_masks()

    def _init_credibility_item_masks(self) -> None:
        n_items = self.dataset.n_items
        device = self.config['device']

        known_cred_mask = torch.zeros(n_items, dtype=torch.bool, device=device)
        known_credible_items = getattr(self.tokenizer, 'known_credible_items',
                                       set())
        for raw_id in known_credible_items:
            item_key = str(raw_id)
            if item_key in self.dataset.item2id:
                known_cred_mask[self.dataset.item2id[item_key]] = True

        known_uncred_mask = torch.zeros(n_items,
                                        dtype=torch.bool,
                                        device=device)
        known_uncredible_items = getattr(self.tokenizer, 'known_uncredible_items',
                                         set())
        for raw_id in known_uncredible_items:
            item_key = str(raw_id)
            if item_key in self.dataset.item2id:
                known_uncred_mask[self.dataset.item2id[item_key]] = True

        unknown_mask = (~known_cred_mask) & (~known_uncred_mask)

        self.item_is_known_credible = known_cred_mask
        self.item_is_known_uncredible = known_uncred_mask
        self.item_is_unknown = unknown_mask
        self.log('[MASK] item groups: '
                 f'known_credible={int(known_cred_mask.sum().item())}, '
                 f'known_uncredible={int(known_uncred_mask.sum().item())}, '
                 f'unknown={int(unknown_mask.sum().item())}')

        self.last_token_cred_mask = str(
            self.config.get('last_token_cred_mask',
                            'known_uncredible_only')).lower()
        self.known_uncredible_mask_mode = str(
            self.config.get('known_uncredible_mask_mode', 'hard')).lower()
        if self.last_token_cred_mask not in (
                'none',
                'known_uncredible_only',
                'known_uncredible_and_unknown',
        ):
            self.log(
                f'[MASK] Unknown last_token_cred_mask={self.last_token_cred_mask!r}, '
                'falling back to known_uncredible_only')
            self.last_token_cred_mask = 'known_uncredible_only'
        if self.known_uncredible_mask_mode not in ('hard', 'soft'):
            self.log(
                f'[MASK] Unknown known_uncredible_mask_mode={self.known_uncredible_mask_mode!r}, '
                'falling back to hard')
            self.known_uncredible_mask_mode = 'hard'
        self.log(f'[MASK] last_token_cred_mask: {self.last_token_cred_mask}')
        self.log(
            f'[MASK] known_uncredible_mask_mode: {self.known_uncredible_mask_mode}')

        # ===== [MOD] configurable credibility-mask token position =====
        # The credibility signal is injected at RQ-VAE residual layer k
        # (`cred_reg_residual_layer` during tokenization), which corresponds to
        # the k-th semantic token. The downstream soft mask must therefore act
        # on the k-th token of the (target / last-history) item, not the last.
        # `cred_mask_token_pos` is 1-indexed; default = n_digit (last token),
        # which reproduces the legacy behavior.
        n_digit = self.tokenizer.n_digit
        self.cred_mask_token_pos = int(
            self.config.get('cred_mask_token_pos', n_digit))
        if self.cred_mask_token_pos < 1 or self.cred_mask_token_pos > n_digit:
            self.log(
                f'[MASK] cred_mask_token_pos={self.cred_mask_token_pos} out of '
                f'range [1, {n_digit}], falling back to {n_digit} (last).')
            self.cred_mask_token_pos = n_digit
        self.log(f'[MASK] cred_mask_token_pos: {self.cred_mask_token_pos} '
                 f'(1-indexed; n_digit={n_digit})')
        # ===== [/MOD] =====

    def _compute_last_token_similarity_weight(
            self, label_item_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if (not self.item_is_known_credible.any()
                or not self.item_is_known_uncredible.any()):
            return None

        token_emb_table = self.cregr.model.transformer.wte.weight
        # ===== [MOD] use the credibility-carrying token position (k-th), which
        # matches the RQ-VAE residual layer used during tokenization, instead
        # of always the last digit. dpos is the 0-indexed digit. =====
        dpos = self.cred_mask_token_pos - 1
        all_last_token_ids = self.item_id2tokens[:, dpos]
        all_last_embs = token_emb_table[all_last_token_ids]

        cred_center = all_last_embs[self.item_is_known_credible].mean(dim=0)
        uncred_center = all_last_embs[self.item_is_known_uncredible].mean(dim=0)

        label_last_token_ids = self.item_id2tokens[label_item_ids, dpos]
        label_last_embs = token_emb_table[label_last_token_ids]
        # ===== [/MOD] =====

        # sim_to_cred = F.cosine_similarity(label_last_embs,
        #                                   cred_center.unsqueeze(0),
        #                                   dim=-1,
        #                                   eps=1e-8)
        # sim_to_uncred = F.cosine_similarity(label_last_embs,
        #                                     uncred_center.unsqueeze(0),
        #                                     dim=-1,
        #                                     eps=1e-8)
        # score = sim_to_cred - sim_to_uncred


        # dist_to_cred = torch.norm(label_last_embs - cred_center, p=2, dim=-1)
        # dist_to_uncred = torch.norm(label_last_embs - uncred_center, p=2, dim=-1)
        # score = dist_to_uncred - dist_to_cred

        dist_to_cred = torch.sum((label_last_embs - cred_center)**2,
                                 dim=-1)
        dist_to_uncred = torch.sum((label_last_embs - uncred_center)**2,
                                   dim=-1)
        score = dist_to_uncred - dist_to_cred
        

        # Keep weight strictly below 1 for soft masking.
        return torch.sigmoid(score).clamp(min=0.0, max=1)

    def _build_last_token_mask_weight(
        self,
        label_item_ids: torch.Tensor,
        no_mask_last_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build per-sample mask weights for the label item's final token.

        Mode ``last_token_cred_mask`` (config):
        - none: no credibility-based last-token loss reweighting (all weights 1).
        - known_uncredible_only: only known-uncredible samples (``no_mask_last_token``):
          * hard: weight 0 on the last token (original behavior).
          * soft: weight = sigmoid(sim_to_cred - sim_to_uncred).
        - known_uncredible_and_unknown: same zeroing for known uncredible, plus unknown
          label items get weight = sigmoid(dist_to_uncred - dist_to_cred) using
          embedding centers of known credible / known uncredible items.
        """
        # Sampling weights are control signals only; keep them out of autograd
        # to avoid inplace/version conflicts in backward.
        with torch.no_grad():
            device = label_item_ids.device
            bsz = label_item_ids.shape[0]
            if self.last_token_cred_mask == 'none':
                return torch.ones(bsz, device=device)
            weight = torch.ones(bsz, device=device)

            if no_mask_last_token is not None:
                known_no_mask = no_mask_last_token.to(device).bool()
                if known_no_mask.any():
                    if self.known_uncredible_mask_mode == 'soft':
                        soft_weight = self._compute_last_token_similarity_weight(
                            label_item_ids)
                        if soft_weight is None:
                            weight[known_no_mask] = 0.0
                        else:
                            weight[known_no_mask] = soft_weight[known_no_mask]
                    else:
                        weight[known_no_mask] = 0.0

            if self.last_token_cred_mask == 'known_uncredible_only':
                return weight.clamp(0.0, 1.0)

            unknown_mask = self.item_is_unknown[label_item_ids]
            if not unknown_mask.any():
                return weight

            similarity_weight = self._compute_last_token_similarity_weight(
                label_item_ids)
            if similarity_weight is None:
                return weight
            unknown_weight = similarity_weight

            replace_mask = unknown_mask & (unknown_weight < 0.42)
            weight[replace_mask] = unknown_weight[replace_mask]
            return weight.clamp(0.0, 1.0)

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(
                self.tokenizer.item2tokens[item])
        return item_id2tokens

    def get_posValidToken(self) -> torch.Tensor:
        posValidTokens = {}

        for item in self.tokenizer.item2tokens:
            cur_tokens = self.tokenizer.item2tokens[item]
            for pos in range(len(cur_tokens)):
                if pos not in posValidTokens.keys():
                    posValidTokens[pos] = set()
                posValidTokens[pos].add(cur_tokens[pos])
        max_token_num = 0

        for pos in posValidTokens:
            posValidTokens[pos] = sorted(list(posValidTokens[pos]))
            max_token_num = max(max_token_num, len(posValidTokens[pos]))

        posValidTokens_pt = torch.zeros((len(posValidTokens), max_token_num),
                                        dtype=torch.long)

        for pos in posValidTokens:
            posValidTokens_pt[
                pos, :len(posValidTokens[pos])] = torch.LongTensor(
                    posValidTokens[pos])

        return posValidTokens_pt

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.cregr.get_input_embeddings().parameters()
            if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def add_item_pos_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        item_seq_len = input_ids.shape[1] // self.tokenizer.n_digit
        item_pos_idx = torch.arange(item_seq_len, device=input_ids.device)

        token_pos_idx = item_pos_idx.repeat_interleave(
            repeats=self.tokenizer.n_digit, dim=0)

        token_pos_emb = self.item_pos_emb(token_pos_idx)
        token_pos_emb = token_pos_emb.unsqueeze(0).expand(
            (input_ids.shape[0], -1, -1))

        input_embeds = self.cregr.model.transformer.wte(input_ids)

        return input_embeds + token_pos_emb

    def cregr_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.add_item_pos_emb(input_ids)
        input_ids = None

        return self.cregr(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )

    def his_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
        last_token_mask_weight: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        masked_input_tokens, masked_indices, p_mask = self.get_masked_seqs(
            input_tokens,
            input_attention_mask,
            last_token_mask_weight=last_token_mask_weight,
        )

        outputs = self.cregr_forward(
            input_ids=masked_input_tokens,
            attention_mask=input_attention_mask.long())

        logits = outputs.logits
        logits = logits / self.temperature

        input_tokens = input_tokens.masked_fill(~input_attention_mask,
                                                self.tokenizer.ignored_label)

        token_loss = self.loss_fct(
            logits[masked_indices],
            input_tokens[masked_indices]) / p_mask[masked_indices]
        loss = token_loss.sum() / (input_tokens.shape[0] *
                                   input_tokens.shape[1])
        outputs.loss = loss

        return outputs

    def target_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
        label_tokens,
        label_attention_mask,
        last_token_mask_weight: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:

        masked_label_tokens, masked_indices, p_mask = self.get_masked_seqs(
            label_tokens,
            label_attention_mask,
            is_target=True,
            last_token_mask_weight=last_token_mask_weight)

        final_inputs_ids = torch.cat([input_tokens, masked_label_tokens],
                                     dim=1)
        final_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)

        outputs = self.cregr_forward(
            input_ids=final_inputs_ids,
            attention_mask=final_attention_mask.long())

        logits = outputs.logits
        logits = logits[:, -masked_label_tokens.shape[1]:, :]

        logits = logits / self.temperature

        label_tokens = label_tokens.masked_fill(~label_attention_mask,
                                                self.tokenizer.ignored_label)

        token_loss = self.loss_fct(
            logits[masked_indices],
            label_tokens[masked_indices]) / p_mask[masked_indices]
        loss = token_loss.sum() / (label_tokens.shape[0] *
                                   label_tokens.shape[1])
        outputs.loss = loss

        return outputs

    def forward(self, batch: dict):
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_tokens = input_tokens.reshape((input_tokens.shape[0], -1))
        input_attention_mask = input_tokens != 0

        assert 'labels' in batch, 'The batch must contain the labels.'
        label_item_ids = batch['labels'].squeeze(1)
        label_tokens = self.item_id2tokens[label_item_ids]
        label_tokens = label_tokens.reshape((label_tokens.shape[0], -1))
        label_attention_mask = torch.ones_like(
            label_tokens, device=label_tokens.device).bool()
        no_mask_last_token = None
        if 'uncredible_last_token_no_mask' in batch:
            no_mask_last_token = batch['uncredible_last_token_no_mask'].view(
                -1).bool()
        last_token_mask_weight = self._build_last_token_mask_weight(
            label_item_ids, no_mask_last_token=no_mask_last_token)

        his_inputs = torch.cat([input_tokens, label_tokens], dim=1)
        his_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)
        his_outputs = self.his_mask_loss(
            his_inputs,
            his_attention_mask,
            last_token_mask_weight=last_token_mask_weight,
        )
        his_loss = his_outputs.loss

        target_outputs = self.target_mask_loss(input_tokens,
                                               input_attention_mask,
                                               label_tokens,
                                               label_attention_mask,
                                               last_token_mask_weight=last_token_mask_weight)
        target_loss = target_outputs.loss

        loss = target_loss + self.his_mask_w * his_loss

        return CreGROutPut(loss=loss,
                              his_mask_loss=his_loss,
                              target_mask_loss=target_loss)

    def get_masked_seqs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_target: bool = False,
        last_token_mask_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_target:
            return self.get_masked_targets(input_ids, attention_mask,
                                           last_token_mask_weight)
        else:
            return self.get_masked_seqs_his(input_ids, attention_mask,
                                            last_token_mask_weight)

    def get_masked_seqs_his(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_token_mask_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape

        masked_num = torch.randint(low=0,
                                   high=self.tokenizer.n_digit + 1,
                                   size=(b, l // self.tokenizer.n_digit),
                                   device=input_ids.device)

        masked_prob = masked_num / self.tokenizer.n_digit
        masked_prob = torch.repeat_interleave(masked_prob,
                                              self.tokenizer.n_digit,
                                              dim=1)

        # Scale last-token mask probability by per-sample weight w:
        # p_last <- w * p_last, where w in [0, 1].
        if last_token_mask_weight is not None and l >= self.tokenizer.n_digit:
            last_token_mask_weight = last_token_mask_weight.to(
                input_ids.device).clamp(0.0, 1.0)
            masked_prob = masked_prob.clone()
            # ===== [MOD] reweight the k-th digit of the LAST item (the label
            # item, appended at the end of the his sequence), not the last
            # digit. pos = l - n_digit + (cred_mask_token_pos - 1). =====
            pos = l - self.tokenizer.n_digit + (self.cred_mask_token_pos - 1)
            masked_prob[:, pos] = masked_prob[:, pos] * last_token_mask_weight
            # ===== [/MOD] =====

        masked_indices = torch.rand((b, l),
                                    device=input_ids.device) < masked_prob

        noisy_batch = torch.where(masked_indices, self.mask_token_id,
                                  input_ids)

        noisy_batch = noisy_batch.masked_fill(~attention_mask, 0)
        masked_indices = masked_indices.masked_fill(~attention_mask, False)

        return noisy_batch, masked_indices, masked_prob

    def get_masked_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_token_mask_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape

        mask_num = torch.randint(1,
                                 self.tokenizer.n_digit + 1, (b, ),
                                 device=input_ids.device)
        p_mask = mask_num / self.tokenizer.n_digit

        rand_scores = torch.rand((b, l), device=input_ids.device)
        rand_scores = rand_scores.masked_fill(attention_mask == 0, 2.0)

        sorted_idx = torch.argsort(rand_scores, dim=1)  # (b, l)

        sort_input_ids = input_ids.clone()
        sort_input_ids = sort_input_ids[torch.arange(b).unsqueeze(1),
                                        sorted_idx]
        sort_mask_pos = torch.arange(
            l, device=input_ids.device)[None, :] < mask_num[:, None]
        sort_input_ids[sort_mask_pos] = self.mask_token_id

        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[torch.arange(b).unsqueeze(1), sorted_idx] = sort_mask_pos
        mask_positions = mask_positions & attention_mask.bool()

        p_mask_matrix = p_mask[:, None].repeat(1, l)
        if last_token_mask_weight is not None and l >= self.tokenizer.n_digit:
            last_token_mask_weight = last_token_mask_weight.to(
                input_ids.device).clamp(0.0, 1.0)
            keep_last = torch.rand((b, ),
                                   device=input_ids.device) < last_token_mask_weight
            # ===== [MOD] act on the k-th digit of the target item, not the
            # last digit. For the target seq l == n_digit, so pos == dpos. =====
            pos = l - self.tokenizer.n_digit + (self.cred_mask_token_pos - 1)
            mask_positions[:, pos] = mask_positions[:, pos] & keep_last
            p_mask_matrix[:, pos] = p_mask_matrix[:, pos] * last_token_mask_weight
            # ===== [/MOD] =====

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = self.mask_token_id
        masked_input_ids = masked_input_ids.masked_fill(~attention_mask, 0)

        return masked_input_ids, mask_positions, p_mask_matrix

    def get_transfer_index(self, input_ids, logits, num_transfer_tokens):
        mask_index = (input_ids == self.mask_token_id)

        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = p.gather(index=x0.unsqueeze(-1), dim=-1).squeeze(-1)

        x0 = torch.where(mask_index, x0, input_ids)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        _, transfer_index = torch.topk(confidence,
                                       k=num_transfer_tokens,
                                       dim=-1)

        return transfer_index

    def generate(self, batch, n_return_sequences=1):
        if batch['split'] == 'val':
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.val_num_beams)
        else:
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.num_beams)

        return outputs

    def beam_search(self, batch, num_return_sequences=1, num_beams=1):
        assert num_beams >= num_return_sequences

        input_ids: torch.Tensor = self.item_id2tokens[batch['input_ids']]
        input_ids = input_ids.reshape((input_ids.shape[0], -1))
        attention_mask: torch.Tensor = input_ids != 0

        batch_size = input_ids.shape[0]
        n_digit = self.tokenizer.n_digit

        # Prepare beam search inputs
        input_ids, attention_mask, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
            input_ids, attention_mask, batch_size, num_beams)

        num_transfer_tokens = n_digit // self.gen_steps
        num_transfered = 0
        for i in range(self.gen_steps):

            if i == (self.gen_steps - 1):
                num_transfer_tokens = n_digit - num_transfered

            outputs = self.cregr_forward(input_ids=input_ids,
                                         attention_mask=attention_mask.long())

            logits = outputs.logits
            logits = logits / self.temperature

            transfer_index = self.get_transfer_index(input_ids, logits,
                                                     num_transfer_tokens)

            for j in range(num_transfer_tokens):
                input_ids, beam_scores = self.beam_search_step(
                    logits, input_ids, transfer_index[:, j], beam_scores,
                    beam_idx_offset, batch_size, num_beams)

            num_transfered += num_transfer_tokens

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        label_ids = input_ids[:, -n_digit:]
        label_ids = label_ids[selection_mask.view(-1), :].reshape(
            -1, num_return_sequences, n_digit)

        return label_ids

    def beam_search_step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size,
        num_beams,
    ):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, vocab_size),
            dim=1).squeeze(1)

        # Calculate log softmax over the last dimension
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - self.tokenizer.n_digit
        valid_tokens = self.posValidTokens[transfer_index - his_len]

        valid_tokens = valid_tokens.masked_fill(valid_tokens == 0,
                                                self.mask_token_id)
        next_token_scores = next_token_scores.gather(index=valid_tokens, dim=1)
        next_token_scores = next_token_scores.masked_fill(
            valid_tokens == self.mask_token_id, -np.inf)
        vocab_size = next_token_scores.shape[-1]

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
            next_token_scores)
        next_token_scores = next_token_scores.view(batch_size,
                                                   num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                    2 * num_beams,
                                                    dim=1,
                                                    largest=True,
                                                    sorted=True)

        next_indices = torch.div(next_tokens,
                                 vocab_size,
                                 rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        input_ids = input_ids[beam_idx + beam_idx_offset]
        transfer_index = transfer_index[beam_idx + beam_idx_offset]

        valid_tokens = self.posValidTokens[transfer_index - his_len]
        beam_next_tokens = valid_tokens.gather(
            index=beam_next_tokens.unsqueeze(1), dim=1).squeeze(1)

        x_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        input_ids[x_index, transfer_index] = beam_next_tokens

        return input_ids, beam_scores

    def prepare_beam_search_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size,
        num_beams,
    ):
        beam_label_ids = torch.ones(
            (batch_size * num_beams, self.tokenizer.n_digit),
            device=self.config['device'],
            dtype=torch.long) * self.mask_token_id
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        all_input_ids = torch.cat([input_ids, beam_label_ids], dim=-1)

        beam_label_attention_mask = torch.ones_like(
            beam_label_ids, device=self.config['device'], dtype=torch.long)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        all_attention_mask = torch.cat(
            [attention_mask, beam_label_attention_mask], dim=-1)

        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams, ))

        beam_idx_offset = torch.arange(
            batch_size, device=self.config['device']).repeat_interleave(
                num_beams) * num_beams

        return all_input_ids, all_attention_mask, initial_beam_scores, beam_idx_offset

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
