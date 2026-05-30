from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.tokenizers.MultiHeadVQVAE.layers import MLPLayers, VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self,
                 n_e_list: List[int],
                 e_dim: int,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters
        if sk_epsilons is None:
            sk_epsilons = [0.0] * self.num_quantizers
        self.sk_epsilons = sk_epsilons

        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e=n_e,
                            e_dim=e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=self.sk_iters)
            for n_e, sk_epsilon in zip(self.n_e_list, self.sk_epsilons)
        ])

    def forward(self, x: torch.Tensor, use_sk=True):
        residual = x
        quantized_sum = torch.zeros_like(x)
        all_indices = []
        all_losses = []

        for quantizer in self.vq_layers:
            x_q, loss, indices = quantizer(residual, use_sk=use_sk)
            quantized_sum = quantized_sum + x_q
            residual = residual - x_q.detach()
            all_indices.append(indices)
            all_losses.append(loss)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return quantized_sum, mean_loss, all_indices


class RQVAE(nn.Module):

    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 title_desc_dim=None,
                 pretrain_dim=64,
                 td_reduce_dim=64,
                 pre_reduce_dim=64,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 # ===== [MOD] new args for decoupled cred-projection head =====
                 cred_proj_dim=None,
                 cred_proj_hidden=None):
                 # ===== [/MOD] =====
        super().__init__()
        if num_emb_list is None:
            num_emb_list = [256, 256, 256, 256]
        if layers is None:
            layers = [512, 256]

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.pretrain_dim = int(pretrain_dim)
        self.td_reduce_dim = int(td_reduce_dim)
        self.pre_reduce_dim = int(pre_reduce_dim)
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        if title_desc_dim is None:
            self.title_desc_dim = self.in_dim - self.pretrain_dim
        else:
            self.title_desc_dim = int(title_desc_dim)
        if self.title_desc_dim <= 0:
            raise ValueError('title_desc_dim must be positive.')
        if self.title_desc_dim + self.pretrain_dim != self.in_dim:
            raise ValueError('title_desc_dim + pretrain_dim must equal in_dim.')

        # Branch-1: concat(title, desc) -> 64
        self.title_desc_reduce = MLPLayers(
            layers=[self.title_desc_dim, self.td_reduce_dim],
            dropout=self.dropout_prob,
            bn=False)
        self.title_desc_norm = nn.LayerNorm(self.td_reduce_dim)
        # Branch-2: pretrain embedding -> 64
        self.pretrain_reduce = MLPLayers(layers=[self.pretrain_dim,
                                                 self.pre_reduce_dim],
                                         dropout=self.dropout_prob,
                                         bn=False)
        self.pretrain_norm = nn.LayerNorm(self.pre_reduce_dim)

        self.encoder_in_dim = self.td_reduce_dim + self.pre_reduce_dim
        self.encode_layer_dims = [self.encoder_in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,
                                 bn=self.bn)

        self.vq = ResidualVectorQuantizer(n_e_list=self.num_emb_list,
                                          e_dim=self.e_dim,
                                          beta=self.beta,
                                          kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters)

        self.decode_layer_dims = [self.e_dim] + self.layers[::-1] + [self.encoder_in_dim]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob,
                                 bn=self.bn)

        # ============================================================
        # [MOD] Decoupled projection head for credibility regularization.
        #
        # Why: the original residual-VQ path forced cred loss onto the
        # encoder via `residual = x_e - sum(q_emb.detach())`, so cred and
        # recon competed head-on for the encoder's parameters. By routing
        # cred through a small head on top of x_e (encoder output, dim
        # = e_dim for RQ-VAE), the codebook is fully untouched by cred
        # grads, and the encoder only receives an attenuated gradient
        # back-propagated through the head.
        #
        # If cred_proj_dim is None or <= 0, the head is disabled and the
        # caller can fall back to the original residual-based query.
        # ============================================================
        if cred_proj_dim is not None and cred_proj_dim > 0:
            # In RQ-VAE the encoder output is [B, e_dim], not e_dim * K.
            encoder_out_dim = self.e_dim
            hidden_dim = (cred_proj_hidden
                          if cred_proj_hidden and cred_proj_hidden > 0
                          else encoder_out_dim)
            self.cred_head = nn.Sequential(
                nn.Linear(encoder_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cred_proj_dim),
            )
        else:
            self.cred_head = None
        # ===== [/MOD] =====

    def _prepare_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(f'Unexpected input shape {tuple(x.shape)} for in_dim={self.in_dim}')
        x_td = x[:, :self.title_desc_dim]
        x_pre = x[:, self.title_desc_dim:]
        x_td = self.title_desc_reduce(x_td)
        x_pre = self.pretrain_reduce(x_pre)
        x_td = self.title_desc_norm(x_td)
        x_pre = self.pretrain_norm(x_pre)
        return torch.cat([x_td, x_pre], dim=-1)

    def encode_inputs(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._prepare_encoder_input(x)
        return self.encoder(x_in)

    # ============================================================
    # [MOD] New helper: project raw inputs into the cred subspace.
    #
    # Returns None when no cred_head is configured (so the caller can
    # detect that and fall back to the legacy residual path).
    # Reuses encode_inputs to honor the dual-branch preprocessing.
    # ============================================================
    def cred_project(self, x: torch.Tensor):
        if self.cred_head is None:
            return None
        x_e = self.encode_inputs(x)
        return self.cred_head(x_e)
    # ===== [/MOD] =====

    def forward(self, x, use_sk=True):
        x_in = self._prepare_encoder_input(x)
        x_e = self.encoder(x_in)
        x_q, vq_loss, indices = self.vq(x_e, use_sk=use_sk)
        out = self.decoder(x_q)

        unused_codebooks = 0
        indices_array = indices.detach().cpu().numpy()
        for i in range(indices.shape[1]):
            cur_used = np.unique(indices_array[:, i]).shape[0]
            unused_codebooks += max(self.num_emb_list[i] - cur_used, 0)

        return out, vq_loss, indices, unused_codebooks

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encode_inputs(xs)
        _, _, indices = self.vq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        xs = self._prepare_encoder_input(xs)
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon, quant_loss
