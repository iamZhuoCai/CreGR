import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_


@torch.no_grad()
def sinkhorn_algorithm_gpu(distances, epsilon, sinkhorn_iterations):
    """GPU-resident fp32 Sinkhorn. Same semantics as the CPU+double version
    below, but avoids two CUDA<->host syncs and the fp64 cast per training
    step. Enable via env var ``VQ_SINKHORN_GPU=1`` to opt in without changing
    the default (reproducible) path.

    Numerical-stability note: subtracts the *global* max of ``-d/eps`` before
    ``exp`` — this is mathematically a no-op because Sinkhorn iterates ratios
    (multiplying every entry by ``exp(-c)`` cancels through the row/col
    normalizations), but bounds the fp32 exponent to avoid overflow. Doing
    per-row max-subtract instead would rescale rows non-uniformly and change
    the optimal-transport solution — empirically that gave <1% argmax
    agreement with the CPU baseline."""
    B, K = distances.shape[0], distances.shape[1]
    log_Q = -distances / epsilon
    log_Q = log_Q - log_Q.amax()        # global, not per-row — see docstring
    Q = torch.exp(log_Q)
    Q = Q / (Q.sum() + 1e-12)
    for _ in range(sinkhorn_iterations):
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        Q = Q / B
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        Q = Q / K
    Q = Q * B
    return Q


class MLPLayers(nn.Module):

    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(
                activation_name))

    return activation


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[
        -1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.003,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e,
                                                1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        elif os.environ.get('VQ_SINKHORN_GPU', '0') == '1':
            # GPU fp32 Sinkhorn. Removes the two GPU<->CPU syncs and the fp64
            # cast that dominate this tokenizer's CPU-bound wall clock. Opt-in
            # to preserve bit-level reproducibility of legacy runs by default.
            d = self.center_distance_for_constraint(d)
            Q = sinkhorn_algorithm_gpu(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn (GPU) returned nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            orig_device = d.device
            d = d.cpu().double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1).to(orig_device)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


class MultiVectorQuantizer(nn.Module):

    def __init__(self,
                 n_e_list,
                 e_dim,
                 sk_epsilons,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e,
                            e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x: torch.Tensor, use_sk=True):
        # NOTE: previously this allocated a (B, num_q, e_dim) zero buffer and
        # filled it via 4 in-place advanced-index writes (`x_q[:, level, :] =
        # x_quant`). Each write triggers an extra autograd/stride dispatch path.
        # A single `torch.cat` is mathematically identical and substantially
        # lighter on the CPU side, which matters because this loop runs once
        # per training step and the workload is CPU-bound (see
        # `cregr-politifact-tokenization-time` note).
        quant_list = []
        all_losses = []
        all_indices = []

        x_chunk = torch.chunk(x, self.num_quantizers, dim=1)
        for level, quantizer in enumerate(self.vq_layers):
            x_quant, loss, indices = quantizer(x_chunk[level], use_sk=use_sk)
            quant_list.append(x_quant)
            all_losses.append(loss)
            all_indices.append(indices)

        x_q = torch.cat(quant_list, dim=1)              # (B, num_q*e_dim)
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices


class MultiHeadVQVAE(nn.Module):

    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100):
        super(MultiHeadVQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [
            self.e_dim * len(self.num_emb_list)
        ]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,
                                 bn=self.bn)

        self.vq = MultiVectorQuantizer(num_emb_list,
                                       e_dim,
                                       beta=self.beta,
                                       kmeans_init=self.kmeans_init,
                                       kmeans_iters=self.kmeans_iters,
                                       sk_epsilons=self.sk_epsilons,
                                       sk_iters=self.sk_iters)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob,
                                 bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, vq_loss, indices = self.vq(x, use_sk=use_sk)
        out = self.decoder(x_q)

        unused_codebooks = 0
        indices_array = indices.detach().cpu().numpy()
        for i in range(indices.shape[1]):
            cur_used = np.unique(indices_array[:, i]).shape[0]

            unused_codebooks += max(self.num_emb_list[i] - cur_used, 0)

        return out, vq_loss, indices, unused_codebooks

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.vq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon, quant_loss
