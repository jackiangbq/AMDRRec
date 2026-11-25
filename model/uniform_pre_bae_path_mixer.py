import torch
from torch import nn


class UniformPreBAEPathMixer(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, node_embeddings, num_paths):

        if node_embeddings.dim() != 2:
            raise ValueError(
                "node_embeddings must have shape "
                "[num_nodes, embedding_dim]."
            )

        if not torch.is_floating_point(node_embeddings):
            raise ValueError(
                "node_embeddings must be floating point."
            )

        if not isinstance(num_paths, int):
            raise TypeError(
                "num_paths must be an integer."
            )

        if num_paths < 1:
            raise ValueError(
                "At least one meta-path is required."
            )

        num_nodes = node_embeddings.size(0)

        router_weights = torch.full(
            size=(num_nodes, num_paths),
            fill_value=1.0 / num_paths,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )

        return router_weights

    def route_graph(
        self,
        graph,
        node_path_weights,
        num_paths,
    ):

        if node_path_weights.dim() != 1:
            raise ValueError(
                "node_path_weights must have shape [num_nodes]."
            )

        if node_path_weights.size(0) != graph.num_nodes():
            raise ValueError(
                "The number of node path weights must match "
                "the number of graph nodes."
            )

        if not torch.is_floating_point(node_path_weights):
            raise ValueError(
                "node_path_weights must be floating point."
            )

        if not isinstance(num_paths, int):
            raise TypeError(
                "num_paths must be an integer."
            )

        if num_paths < 1:
            raise ValueError(
                "At least one meta-path is required."
            )

        source_nodes, destination_nodes = graph.edges()

        if destination_nodes.device != node_path_weights.device:
            raise ValueError(
                "The graph and node path weights must be "
                "on the same device."
            )

        routed_graph = graph.local_var()

        if "edge_weight" in graph.edata:
            original_edge_weight = graph.edata["edge_weight"]
        else:
            original_edge_weight = torch.ones(
                graph.num_edges(),
                dtype=node_path_weights.dtype,
                device=node_path_weights.device,
            )

        if original_edge_weight.dim() != 1:
            raise ValueError(
                "graph.edata['edge_weight'] must have shape "
                "[num_edges]."
            )

        if original_edge_weight.device != node_path_weights.device:
            raise ValueError(
                "Edge weights and node path weights must be "
                "on the same device."
            )

        routing_multiplier = (
            num_paths
            * node_path_weights[destination_nodes]
        )

        routed_graph.edata["edge_weight"] = (
            original_edge_weight
            * routing_multiplier
        )

        return routed_graph
