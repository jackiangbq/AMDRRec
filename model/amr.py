"""Adaptive Meta-Path Routing (AMR), manuscript Equations (2)--(7)."""

import torch
from torch import nn


class UniformMetaPathRouter(nn.Module):

    def forward(self, node_embeddings, num_paths):
        if node_embeddings.dim() != 2:
            raise ValueError("node_embeddings must be two-dimensional.")
        if not torch.is_floating_point(node_embeddings):
            raise ValueError("node_embeddings must be floating point.")
        if not isinstance(num_paths, int) or num_paths < 1:
            raise ValueError("num_paths must be a positive integer.")
        return torch.full(
            (node_embeddings.size(0), num_paths),
            1.0 / num_paths,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )

    def route_graph(self, graph, node_path_weights, num_paths):

        if node_path_weights.dim() != 1:
            raise ValueError("node_path_weights must have shape [num_nodes].")
        if node_path_weights.size(0) != graph.num_nodes():
            raise ValueError("Graph nodes and path weights must have equal size.")
        _, destination_nodes = graph.edges()
        routed_graph = graph.local_var()
        if "edge_weight" in graph.edata:
            original_edge_weight = graph.edata["edge_weight"]
        else:
            original_edge_weight = torch.ones(
                graph.num_edges(),
                dtype=node_path_weights.dtype,
                device=node_path_weights.device,
            )
        routed_graph.edata["edge_weight"] = (
            original_edge_weight
            * num_paths
            * node_path_weights[destination_nodes]
        )
        return routed_graph


class AdaptiveMetaPathRouter(UniformMetaPathRouter):

    def __init__(self, num_paths, top_k=2):
        super().__init__()
        if not isinstance(num_paths, int) or num_paths < 1:
            raise ValueError("num_paths must be a positive integer.")
        if not isinstance(top_k, int) or not 1 <= top_k <= num_paths:
            raise ValueError("top_k must be between 1 and num_paths.")
        self.num_paths = num_paths
        self.top_k = top_k

        self.path_logits = nn.Parameter(torch.zeros(num_paths))

    def forward(self, node_embeddings, num_paths):
        super().forward(node_embeddings, num_paths)
        if num_paths != self.num_paths:
            raise ValueError("Input path count differs from configured num_paths.")
        if self.path_logits.device != node_embeddings.device:
            raise ValueError("Router logits and node embeddings must share a device.")
        if self.path_logits.dtype != node_embeddings.dtype:
            raise ValueError("Router logits and node embeddings must share a dtype.")
        global_path_weights = torch.softmax(self.path_logits, dim=0)
        return global_path_weights.unsqueeze(0).expand(node_embeddings.size(0), -1)

    def get_hard_topk_route(self, node_embeddings, num_paths):

        global_path_weights = self.forward(node_embeddings, num_paths)[0]
        ranked_indices = sorted(
            range(num_paths),
            key=lambda index: (
                -float(global_path_weights[index].detach().cpu()),
                index,
            ),
        )
        topk_indices = torch.tensor(
            ranked_indices[: self.top_k],
            dtype=torch.long,
            device=global_path_weights.device,
        )
        hard_mask = torch.zeros_like(global_path_weights)
        hard_mask.scatter_(0, topk_indices, 1.0)
        straight_through_mask = (
            hard_mask + global_path_weights - global_path_weights.detach()
        )
        return {
            "global_path_weights": global_path_weights,
            "topk_indices": topk_indices,
            "hard_mask": hard_mask,
            "straight_through_mask": straight_through_mask,
            "selected_gate_weights": straight_through_mask.index_select(
                0, topk_indices
            ),
        }


AMR = AdaptiveMetaPathRouter
