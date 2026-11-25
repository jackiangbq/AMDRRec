import math

import torch
from torch import nn


class UniformMeanRouter(nn.Module):

    def __init__(
        self,
        residual_fusion_gamma=1e-5,
        fusion_scale=None,
    ):
        super().__init__()

        if fusion_scale is not None:
            residual_fusion_gamma = fusion_scale
        residual_fusion_gamma = float(residual_fusion_gamma)

        if not math.isfinite(residual_fusion_gamma):
            raise ValueError(
                "residual_fusion_gamma must be finite."
            )

        if residual_fusion_gamma < 0.0:
            raise ValueError(
                "residual_fusion_gamma must be non-negative."
            )

        self.residual_fusion_gamma = residual_fusion_gamma
        self.fusion_scale = residual_fusion_gamma

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

        expected_base_shape = (
            num_nodes,
            embedding_dim,
        )

        if tuple(base_embeddings.shape) != expected_base_shape:
            raise ValueError(
                "base_embeddings and path_embeddings "
                "have incompatible shapes."
            )

        router_weights = torch.full(
            size=(num_nodes, num_paths),
            fill_value=1.0 / num_paths,
            dtype=path_embeddings.dtype,
            device=path_embeddings.device,
        )

        weighted_path_embeddings = torch.sum(
            router_weights.unsqueeze(-1)
            * path_embeddings,
            dim=1,
        )


        compensated_path_embeddings = (
            num_paths
            * self.residual_fusion_gamma
            * weighted_path_embeddings
        )

        fused_embeddings = (
            base_embeddings
            + compensated_path_embeddings
        )

        return fused_embeddings, router_weights
