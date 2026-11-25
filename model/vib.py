
import math

import torch
import torch.nn.functional as F
from torch import nn


class VariationalInformationBottleneck(nn.Module):


    def __init__(self, input_dim, latent_dim, beta, recommendation_weight):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("vib_latent_dim must be at least 1.")
        self.beta = float(beta)
        self.recommendation_weight = float(recommendation_weight)
        self.latent_dim = int(latent_dim)
        posterior_parameter_dim = 2 * self.latent_dim
        self.user_compressor = nn.Linear(input_dim, posterior_parameter_dim)
        self.item_compressor = nn.Linear(input_dim, posterior_parameter_dim)
        self.last_compression_loss = None
        self.last_recommendation_loss = None
        self.last_user_latents = None
        self.last_item_latents = None

    def _posterior(self, encoded):
        mu = encoded[:, :self.latent_dim]
        std = F.softplus(
            encoded[:, self.latent_dim:],
            beta=1,
        ) + 1e-8
        return mu, std

    def _sample(self, mu, std):
        if self.training:
            return mu + std * torch.randn_like(std)
        return mu

    @staticmethod
    def _kl_to_unit_gaussian(mu, std):

        return -0.5 / math.log(2.0) * (
            1 + 2 * std.log() - mu.pow(2) - std.pow(2)
        ).sum(1).mean()

    @staticmethod
    def _bpr_loss(user_latent, positive_latent, negative_latent):
        positive_score = (user_latent * positive_latent).sum(dim=-1)
        negative_score = (user_latent * negative_latent).sum(dim=-1)
        return -F.logsigmoid(positive_score - negative_score).mean()

    def _encode_paths(self, representations, compressor):
        if representations.dim() != 3:
            raise ValueError(
                "VIB representations must have shape "
                "[num_entities, num_paths, embedding_dim]."
            )
        num_entities, num_paths, embedding_dim = representations.shape
        encoded = compressor(
            representations.reshape(num_entities * num_paths, embedding_dim)
        )
        mu, std = self._posterior(encoded)
        latent = self._sample(mu, std)
        return (
            mu.reshape(num_entities, num_paths, self.latent_dim),
            std.reshape(num_entities, num_paths, self.latent_dim),
            latent.reshape(num_entities, num_paths, self.latent_dim),
        )

    def forward(
        self,
        user_path_representations,
        item_path_representations,
        user_indices,
        positive_item_indices,
        negative_item_indices,
    ):
        if (
            user_path_representations.size(1)
            != item_path_representations.size(1)
        ):
            raise ValueError("User and item VIB path counts must match.")

        user_mu, user_std, user_latents = self._encode_paths(
            user_path_representations,
            self.user_compressor,
        )
        item_mu, item_std, item_latents = self._encode_paths(
            item_path_representations,
            self.item_compressor,
        )

        compression_terms = []
        recommendation_terms = []
        for path_index in range(user_path_representations.size(1)):
            compression_terms.append(
                self._kl_to_unit_gaussian(
                    user_mu[:, path_index, :],
                    user_std[:, path_index, :],
                )
                + self._kl_to_unit_gaussian(
                    item_mu[:, path_index, :],
                    item_std[:, path_index, :],
                )
            )
            recommendation_terms.append(
                self._bpr_loss(
                    user_latents[user_indices.long(), path_index, :],
                    item_latents[positive_item_indices.long(), path_index, :],
                    item_latents[negative_item_indices.long(), path_index, :],
                )
            )

        compression_loss = torch.stack(compression_terms).sum()
        recommendation_loss = torch.stack(recommendation_terms).sum()
        self.last_compression_loss = compression_loss
        self.last_recommendation_loss = recommendation_loss
        self.last_user_latents = user_latents
        self.last_item_latents = item_latents

        return (
            self.beta * compression_loss
            + self.recommendation_weight * recommendation_loss
        )


VIB = VariationalInformationBottleneck
