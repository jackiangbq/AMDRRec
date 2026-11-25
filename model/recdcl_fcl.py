

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecDCLFeatureObjective(nn.Module):


    def __init__(
        self,
        embedding_dim: int,
        bt_offdiag_coeff: float = 0.01,
        uuii_coeff: float = 0.2,
        poly_a: float = 1.0,
        poly_c: float = 1e-7,
        poly_degree: int = 4,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}."
            )
        if bt_offdiag_coeff < 0.0:
            raise ValueError(
                "bt_offdiag_coeff must be non-negative, "
                f"got {bt_offdiag_coeff}."
            )
        if uuii_coeff < 0.0:
            raise ValueError(
                f"uuii_coeff must be non-negative, got {uuii_coeff}."
            )
        if poly_degree <= 0:
            raise ValueError(
                f"poly_degree must be positive, got {poly_degree}."
            )

        self.embedding_dim = int(embedding_dim)
        self.bt_offdiag_coeff = float(bt_offdiag_coeff)
        self.uuii_coeff = float(uuii_coeff)
        self.poly_a = float(poly_a)
        self.poly_c = float(poly_c)
        self.poly_degree = int(poly_degree)


        self.projector = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
                bias=False,
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
                bias=False,
            ),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
                bias=False,
            ),
        )


        self.correlation_bn = nn.BatchNorm1d(
            self.embedding_dim,
            affine=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    @staticmethod
    def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:

        rows, columns = matrix.shape

        if rows != columns:
            raise ValueError(
                "The correlation matrix must be square, "
                f"got shape {tuple(matrix.shape)}."
            )

        return (
            matrix.flatten()[:-1]
            .view(rows - 1, rows + 1)[:, 1:]
            .flatten()
        )

    def _validate_embeddings(
        self,
        user_embeddings: torch.Tensor,
        positive_item_embeddings: torch.Tensor,
    ) -> None:
        if user_embeddings.ndim != 2:
            raise ValueError(
                "user_embeddings must have shape [batch, dim], "
                f"got {tuple(user_embeddings.shape)}."
            )

        if positive_item_embeddings.ndim != 2:
            raise ValueError(
                "positive_item_embeddings must have shape [batch, dim], "
                f"got {tuple(positive_item_embeddings.shape)}."
            )

        if user_embeddings.shape != positive_item_embeddings.shape:
            raise ValueError(
                "User and positive-item embedding shapes must match, "
                f"got {tuple(user_embeddings.shape)} and "
                f"{tuple(positive_item_embeddings.shape)}."
            )

        if user_embeddings.size(0) <= 1:
            raise ValueError(
                "RecDCL-FCL requires a batch size greater than one."
            )

        if user_embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{self.embedding_dim}, got {user_embeddings.size(1)}."
            )

    def ui_bt_loss(
        self,
        user_embeddings: torch.Tensor,
        positive_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        normalized_users = F.normalize(
            user_embeddings,
            dim=-1,
        )
        normalized_items = F.normalize(
            positive_item_embeddings,
            dim=-1,
        )

        projected_users = self.projector(normalized_users)
        projected_items = self.projector(normalized_items)

        standardized_users = self.correlation_bn(projected_users)
        standardized_items = self.correlation_bn(projected_items)

        batch_size = standardized_users.size(0)

        correlation = (
            standardized_users.T @ standardized_items
        ) / batch_size

        diagonal_loss = (
            (torch.diagonal(correlation) - 1.0)
            .pow(2)
            .sum()
            / self.embedding_dim
        )

        off_diagonal_loss = (
            self.off_diagonal(correlation)
            .pow(2)
            .sum()
            / self.embedding_dim
        )

        return (
            diagonal_loss
            + self.bt_offdiag_coeff * off_diagonal_loss
        )

    def polynomial_feature_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        normalized_embeddings = F.normalize(
            embeddings,
            dim=-1,
        )

        projected_embeddings = self.projector(
            normalized_embeddings
        )
        standardized_embeddings = self.correlation_bn(
            projected_embeddings
        )

        feature_gram = (
            standardized_embeddings.T
            @ standardized_embeddings
        )

        polynomial_matrix = (
            self.poly_a * feature_gram + self.poly_c
        ).pow(self.poly_degree)

        return polynomial_matrix.mean().log()

    def forward(
        self,
        user_embeddings: torch.Tensor,
        positive_item_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._validate_embeddings(
            user_embeddings,
            positive_item_embeddings,
        )

        uibt_loss = self.ui_bt_loss(
            user_embeddings,
            positive_item_embeddings,
        )


        if self.uuii_coeff == 0.0:
            uuii_loss = uibt_loss.new_zeros(())
        else:
            user_uuii_loss = self.polynomial_feature_loss(
                user_embeddings
            )
            item_uuii_loss = self.polynomial_feature_loss(
                positive_item_embeddings
            )

            uuii_loss = 0.5 * (
                user_uuii_loss + item_uuii_loss
            )

        fcl_loss = (
            uibt_loss
            + self.uuii_coeff * uuii_loss
        )

        return fcl_loss, uibt_loss, uuii_loss