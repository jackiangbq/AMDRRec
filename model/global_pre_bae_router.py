import torch
from torch import nn

from .uniform_pre_bae_path_mixer import (
    UniformPreBAEPathMixer,
)


class GlobalPreBAERouter(UniformPreBAEPathMixer):

    def __init__(self, num_paths, top_k=2):
        super().__init__()

        if not isinstance(num_paths, int):
            raise TypeError(
                "num_paths must be an integer."
            )

        if num_paths < 1:
            raise ValueError(
                "At least one meta-path is required."
            )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k < 1 or top_k > num_paths:
            raise ValueError(
                "top_k must be between 1 and num_paths."
            )

        self.num_paths = num_paths
        self.top_k = top_k


        self.path_logits = nn.Parameter(
            torch.zeros(num_paths)
        )

    def forward(self, node_embeddings, num_paths):

        super().forward(
            node_embeddings=node_embeddings,
            num_paths=num_paths,
        )

        if num_paths != self.num_paths:
            raise ValueError(
                "The input num_paths does not match "
                "the configured number of meta-paths."
            )

        if (
            self.path_logits.device
            != node_embeddings.device
        ):
            raise ValueError(
                "The Router parameters and node embeddings "
                "must be on the same device."
            )

        if (
            self.path_logits.dtype
            != node_embeddings.dtype
        ):
            raise ValueError(
                "The Router parameters and node embeddings "
                "must have the same dtype."
            )

        # [K]
        global_path_weights = torch.softmax(
            self.path_logits,
            dim=0,
        )

        # [K] -> [N, K]
        router_weights = (
            global_path_weights
            .unsqueeze(0)
            .expand(
                node_embeddings.size(0),
                -1,
            )
        )

        return router_weights

    def get_hard_topk_route(
        self,
        node_embeddings,
        num_paths,
    ):

        router_weights = self.forward(
            node_embeddings=node_embeddings,
            num_paths=num_paths,
        )


        global_path_weights = router_weights[0]

         ranked_indices = sorted(
            range(num_paths),
            key=lambda index: (
                -float(
                    global_path_weights[index]
                    .detach()
                    .cpu()
                ),
                index,
            ),
        )

        topk_indices = torch.tensor(
            ranked_indices[:self.top_k],
            dtype=torch.long,
            device=global_path_weights.device,
        )

        hard_mask = torch.zeros_like(
            global_path_weights
        )
        hard_mask.scatter_(
            dim=0,
            index=topk_indices,
            value=1.0,
        )

        straight_through_mask = (
            hard_mask
            + global_path_weights
            - global_path_weights.detach()
        )

        selected_gate_weights = (
            straight_through_mask.index_select(
                dim=0,
                index=topk_indices,
            )
        )

        return {
            "global_path_weights": global_path_weights,
            "topk_indices": topk_indices,
            "hard_mask": hard_mask,
            "straight_through_mask": straight_through_mask,
            "selected_gate_weights": selected_gate_weights,
        }    
