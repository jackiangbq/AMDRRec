import math

import torch
from torch import nn


class GlobalMetaPathRouter(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim=64,
        fusion_scale=1e-5,
        router_residual_scale=0.1,
    ):
        super().__init__()

        embedding_dim = int(embedding_dim)
        hidden_dim = int(hidden_dim)
        fusion_scale = float(fusion_scale)
        router_residual_scale = float(router_residual_scale)
        if embedding_dim < 1:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not math.isfinite(fusion_scale):
            raise ValueError(
                "fusion_scale must be finite."
            )

        if fusion_scale < 0.0:
            raise ValueError(
                "fusion_scale must be non-negative.")

        if not math.isfinite(router_residual_scale):
            raise ValueError(
                "router_residual_scale must be finite.")

        if router_residual_scale < 0.0:
            raise ValueError(
                "router_residual_scale must be non-negative.")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fusion_scale = fusion_scale
        self.router_residual_scale = (router_residual_scale)
        self.feature_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
            bias=True,
        )
        self.score_projection = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

        nn.init.xavier_uniform_(
            self.feature_projection.weight,
            gain=1.0,
        )
        nn.init.zeros_(
            self.feature_projection.bias
        )

        nn.init.zeros_(
            self.score_projection.weight
        )

    def forward(
        self,
        base_embeddings,
        path_embeddings,
    ):

        if base_embeddings.dim() != 2:
            raise ValueError(
                "base_embeddings must have shape "
                "[num_nodes, embedding_dim]."
            )

        if path_embeddings.dim() != 3:
            raise ValueError(
                "path_embeddings must have shape "
                "[num_nodes, num_paths, embedding_dim]."
            )

        if not torch.is_floating_point(base_embeddings):
            raise ValueError(
                "base_embeddings must be floating point."
            )

        if not torch.is_floating_point(path_embeddings):
            raise ValueError(
                "path_embeddings must be floating point."
            )

        if base_embeddings.device != path_embeddings.device:
            raise ValueError(
                "base_embeddings and path_embeddings "
                "must be on the same device."
            )

        if base_embeddings.dtype != path_embeddings.dtype:
            raise ValueError(
                "base_embeddings and path_embeddings "
                "must have the same dtype."
            )

        num_nodes = path_embeddings.size(0)
        num_paths = path_embeddings.size(1)
        embedding_dim = path_embeddings.size(2)

        if num_paths < 1:
            raise ValueError(
                "At least one path is required."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "The path embedding dimension does not "
                "match the configured embedding_dim."
            )

        expected_base_shape = (
            num_nodes,
            embedding_dim,
        )

        if tuple(base_embeddings.shape) != expected_base_shape:
            raise ValueError(
                "base_embeddings and path_embeddings "
                "have incompatible shapes."
            )

        # [N, K, D] -> [N, K, hidden_dim]
        projected_path_embeddings = torch.tanh(
            self.feature_projection(
                path_embeddings
            )
        )

        global_path_summaries = torch.mean(
            projected_path_embeddings,
            dim=0,
        )

        # [K, H] -> [K]
        global_path_scores = self.score_projection(
            global_path_summaries
        ).squeeze(-1)

        global_path_weights = torch.softmax(
            global_path_scores,
            dim=0,
        )

        router_weights = (
            global_path_weights
            .unsqueeze(0)
            .expand(num_nodes, -1)
        )

        # weighted_path_embeddings = torch.sum(
        #     router_weights.unsqueeze(-1)
        #     * path_embeddings,
        #     dim=1,
        # )

        # compensated_path_embeddings = (
        #     num_paths
        #     * self.fusion_scale
        #     * weighted_path_embeddings
        # )

        # fused_embeddings = (
        #     base_embeddings
        #     + compensated_path_embeddings
        # )

        # return fused_embeddings, router_weights
        # E + fusion_scale * sum_k(S_k)
        fixed_path_contribution = (self.fusion_scale * torch.sum(path_embeddings,dim=1,))

        fixed_fused_embeddings = (base_embeddings + fixed_path_contribution)

        # S_k = E + H_k，因此H_k = S_k - E。
        raw_path_residuals = (path_embeddings - base_embeddings.unsqueeze(1))

        uniform_path_weight = 1.0 / num_paths
        centered_router_weights = (router_weights - uniform_path_weight)

        router_residual = (self.router_residual_scale * torch.sum(centered_router_weights.unsqueeze(-1)
                * raw_path_residuals,dim=1,))

        fused_embeddings = (fixed_fused_embeddings + router_residual)

        return fused_embeddings, router_weights