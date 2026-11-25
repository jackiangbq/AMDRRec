from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch
import dgl


class Autoencoder(nn.Module):
    def __init__(self,in_dim,hidden_dim,enc_num_layer,dec_num_layer,mask_rate,remask_rate,num_remasking,adrm_mode="adaptive_dual_recon",
        attr_mask_rate_start=0.1,attr_mask_rate_end=0.3,attr_sce_alpha=2.0,edge_mask_rate_start=0.1,edge_mask_rate_end=0.3,bae_mode=None,):
        super(Autoencoder, self).__init__()
        if bae_mode is not None:
            adrm_mode = bae_mode
        self.adrm_mode = str(adrm_mode).lower()
        self.bae_mode = self.adrm_mode

        if self.adrm_mode not in {"v1","adaptive_dual_recon",}:
            raise ValueError("Unsupported ADRM mode: "  f"{self.adrm_mode}")
        self.attr_mask_rate_start = float(attr_mask_rate_start)
        self.attr_mask_rate_end = float(attr_mask_rate_end)
        self.attr_sce_alpha = float(attr_sce_alpha)
        self.edge_mask_rate_start = float(
            edge_mask_rate_start
        )
        self.edge_mask_rate_end = float(
            edge_mask_rate_end
        )

        if not (
            0.0
            <= self.attr_mask_rate_start
            <= self.attr_mask_rate_end
            < 1.0
        ):
            raise ValueError(
                "Attribute mask rates must satisfy "
                "0 <= start <= end < 1, but got "
                f"start={self.attr_mask_rate_start}, "
                f"end={self.attr_mask_rate_end}."
            )

        if not (
            0.0
            <= self.edge_mask_rate_start
            <= self.edge_mask_rate_end
            < 1.0
        ):
            raise ValueError(
                "Edge mask rates must satisfy "
                "0 <= start <= end < 1, but got "
                f"start={self.edge_mask_rate_start}, "
                f"end={self.edge_mask_rate_end}."
            )
        
        if self.attr_sce_alpha <= 0.0:
            raise ValueError(
                "attr_sce_alpha must be greater "
                "than 0, but got "
                f"{self.attr_sce_alpha}."
            )        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # encoder
        for i in range(enc_num_layer):
            self.encoder.append(GraphConv(in_dim, hidden_dim, bias=False, weight=False,
                                          allow_zero_in_degree=True))

        # decoder
        for i in range(dec_num_layer):
            self.decoder.append(GraphConv(in_dim, hidden_dim, bias=False, weight=False,
                                          allow_zero_in_degree=True))

        # random_mask
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoder_to_decoder = nn.Linear(in_dim, hidden_dim, bias=False)
        self.decoder_to_contrastive = nn.Linear(in_dim, hidden_dim, bias=False)
        self.last_attr_loss = None
        self.last_attr_valid_count = 0        
        self.last_attr_mask_rate = None
        self.last_edge_loss = None
        self.last_masked_edge_count = 0
        self.last_edge_mask_rate = None
        self.last_encoder_embedding = None
        self.last_intent_embedding = None
        self.last_bottleneck_embedding = None

        self.reset_parameters_for_token()
        self.edge_mask_scorer = None
        self.edge_reconstruction_scorer = None
        if self.adrm_mode == "adaptive_dual_recon":
            self.edge_mask_scorer = nn.Sequential(
                nn.Linear(
                    2 * in_dim + 1,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    1,
                ),
            )

            self.edge_reconstruction_scorer = (
                nn.Sequential(
                    nn.Linear(
                        2 * hidden_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_dim,
                        1,
                    ),
                )
            )

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def get_dynamic_attr_mask_rate(
        self,
        epoch,
        total_epochs,
    ):

        if epoch is None or total_epochs is None:
            return self.attr_mask_rate_start

        total_epochs = int(total_epochs)

        if total_epochs <= 1:
            return self.attr_mask_rate_start

        progress = float(epoch) / float(
            total_epochs - 1
        )
        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            self.attr_mask_rate_start
            + progress
            * (
                self.attr_mask_rate_end
                - self.attr_mask_rate_start
            )
        )
    def get_dynamic_edge_mask_rate(
        self,
        epoch,
        total_epochs,
    ):

        if epoch is None or total_epochs is None:
            return self.edge_mask_rate_start

        total_epochs = int(total_epochs)

        if total_epochs <= 1:
            return self.edge_mask_rate_start

        progress = float(epoch) / float(
            total_epochs - 1
        )
        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            self.edge_mask_rate_start
            + progress
            * (
                self.edge_mask_rate_end
                - self.edge_mask_rate_start
            )
        )

    def adaptive_metapath_edge_mask(
        self,
        g,
        x,
        epoch,
        total_epochs,
    ):

        if self.edge_mask_scorer is None:
            raise RuntimeError(
                "edge_mask_scorer is only "
                "available in adaptive_dual_recon mode."
            )

        current_mask_rate = (
            self.get_dynamic_edge_mask_rate(
                epoch=epoch,
                total_epochs=total_epochs,
            )
        )

        source_nodes, target_nodes = g.edges()
        num_edges = g.num_edges()

        if "edge_weight" in g.edata:
            base_edge_weight = g.edata[
                "edge_weight"
            ]
        else:
            base_edge_weight = torch.ones(
                num_edges,
                device=x.device,
                dtype=x.dtype,
            )

        base_edge_weight = (
            base_edge_weight.to(
                device=x.device,
                dtype=x.dtype,
            ).reshape(-1)
        )

        if num_edges == 0:
            empty_edges = torch.empty(
                0,
                dtype=torch.long,
                device=x.device,
            )

            empty_scores = torch.empty(
                0,
                dtype=x.dtype,
                device=x.device,
            )

            return (
                base_edge_weight,
                empty_edges,
                empty_edges,
                current_mask_rate,
                empty_scores,
            )

        edge_features = torch.cat(
            [
                x[source_nodes],
                x[target_nodes],
                base_edge_weight.unsqueeze(-1),
            ],
            dim=-1,
        )

        mask_logits = self.edge_mask_scorer(
            edge_features
        ).squeeze(-1)

        mask_probabilities = torch.sigmoid(
            mask_logits
        )

        active_edges = torch.nonzero(
            base_edge_weight.detach() > 0.0,
            as_tuple=False,
        ).flatten()

        num_mask_edges = int(
            current_mask_rate
            * active_edges.numel()
        )

        if num_mask_edges > 0:
            active_mask_scores = (
                mask_probabilities[
                    active_edges
                ]
            )

            relative_mask_indices = torch.topk(
                active_mask_scores,
                k=num_mask_edges,
                largest=True,
                sorted=False,
            ).indices

            mask_edges = active_edges[
                relative_mask_indices
            ]
        else:
            mask_edges = torch.empty(
                0,
                dtype=torch.long,
                device=x.device,
            )

        hard_keep_gate = torch.ones_like(
            mask_probabilities
        )
        hard_keep_gate[mask_edges] = 0.0

        soft_keep_gate = (
            1.0 - mask_probabilities
        )

        straight_through_keep_gate = (
            hard_keep_gate.detach()
            - soft_keep_gate.detach()
            + soft_keep_gate
        )

        masked_edge_weight = (
            base_edge_weight
            * straight_through_keep_gate
        )

        keep_edges = active_edges[
            hard_keep_gate[
                active_edges
            ].bool()
        ]

        return (
            masked_edge_weight,
            mask_edges,
            keep_edges,
            current_mask_rate,
            mask_probabilities,
        )

    def dynamic_attribute_mask(
        self,
        g,
        x,
        epoch,
        total_epochs,
    ):

        current_mask_rate = (
            self.get_dynamic_attr_mask_rate(
                epoch=epoch,
                total_epochs=total_epochs,
            )
        )

        num_nodes = x.shape[0]

        if g.num_nodes() != num_nodes:
            raise RuntimeError(
                "Attribute-mask graph/node mismatch: "
                f"graph has {g.num_nodes()} nodes, "
                f"but x has {num_nodes} rows."
            )

        _, destination_nodes = g.edges()

        if "edge_weight" in g.edata:
            effective_edge_weight = (
                g.edata["edge_weight"]
                .detach()
                .reshape(-1)
                .to(
                    device=x.device,
                    dtype=x.dtype,
                )
                .abs()
            )
        else:
            effective_edge_weight = torch.ones(
                g.num_edges(),
                device=x.device,
                dtype=x.dtype,
            )

        effective_in_degree = torch.zeros(
            num_nodes,
            device=x.device,
            dtype=x.dtype,
        )

        effective_in_degree.index_add_(
            0,
            destination_nodes.to(x.device),
            effective_edge_weight,
        )

        reconstructable_nodes = (
            effective_in_degree > 0.0
        )


        permutation = torch.randperm(
            num_nodes,
            device=x.device,
        )

        reconstructable_permutation = (
            permutation[
                reconstructable_nodes[
                    permutation
                ]
            ]
        )

        num_reconstructable_nodes = int(
            reconstructable_permutation.numel()
        )

        num_mask_nodes = int(
            current_mask_rate
            * num_reconstructable_nodes
        )

        mask_nodes = (
            reconstructable_permutation[
                :num_mask_nodes
            ]
        )

        keep_node_mask = torch.ones(
            num_nodes,
            dtype=torch.bool,
            device=x.device,
        )

        keep_node_mask[mask_nodes] = False

        keep_nodes = torch.nonzero(
            keep_node_mask,
            as_tuple=False,
        ).flatten()

        masked_x = x.clone()
        masked_x[mask_nodes] = 0.0
        masked_x[mask_nodes] += (
            self.enc_mask_token
        )

        target_x = x.detach()

        return (
            masked_x,
            target_x,
            mask_nodes,
            keep_nodes,
            current_mask_rate,
        )

    def attribute_reconstruction_loss(
        self,
        prediction,
        target,
        mask_nodes,
    ):

        if mask_nodes.numel() == 0:
            self.last_attr_valid_count = 0
            return prediction.sum() * 0.0

        masked_prediction = prediction[
            mask_nodes
        ]
        masked_target = target[
            mask_nodes
        ]

        normalization_epsilon = 1e-12

        prediction_norm = (
            torch.linalg.vector_norm(
                masked_prediction,
                ord=2,
                dim=-1,
            )
        )

        target_norm = (
            torch.linalg.vector_norm(
                masked_target,
                ord=2,
                dim=-1,
            )
        )

        valid_reconstruction = (
            prediction_norm
            > normalization_epsilon
        ) & (
            target_norm
            > normalization_epsilon
        )

        self.last_attr_valid_count = int(
            valid_reconstruction.sum().item()
        )

        if self.last_attr_valid_count == 0:
            return prediction.sum() * 0.0

        masked_prediction = (
            masked_prediction[
                valid_reconstruction
            ]
        )

        masked_target = (
            masked_target[
                valid_reconstruction
            ]
        )

        masked_prediction = (
            torch.nn.functional.normalize(
                masked_prediction,
                p=2,
                dim=-1,
                eps=normalization_epsilon,
            )
        )

        masked_target = (
            torch.nn.functional.normalize(
                masked_target,
                p=2,
                dim=-1,
                eps=normalization_epsilon,
            )
        )

        cosine_error = 1.0 - (
            masked_prediction
            * masked_target
        ).sum(dim=-1)

        attr_loss = cosine_error.pow(
            self.attr_sce_alpha
        ).mean()

        return attr_loss

    def score_reconstructed_edges(
        self,
        node_embeddings,
        source_nodes,
        target_nodes,
    ):

        if self.edge_reconstruction_scorer is None:
            raise RuntimeError(
                "edge_reconstruction_scorer is "
                "only available in "
                "adaptive_dual_recon mode."
            )

        if (
            source_nodes.numel()
            != target_nodes.numel()
        ):
            raise ValueError(
                "source_nodes and target_nodes "
                "must contain the same number "
                "of indices."
            )

        if source_nodes.numel() == 0:
            return node_embeddings.new_empty(
                (0,)
            )

        pair_features = torch.cat(
            [
                node_embeddings[source_nodes],
                node_embeddings[target_nodes],
            ],
            dim=-1,
        )

        edge_logits = (
            self.edge_reconstruction_scorer(
                pair_features
            ).squeeze(-1)
        )

        return edge_logits

    def sample_structure_negative_edges(
        self,
        g,
        num_samples,
        device,
    ):

        num_samples = int(
            num_samples
        )

        if num_samples <= 0:
            empty_edges = torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )
            return empty_edges, empty_edges

        (
            negative_source_nodes,
            negative_target_nodes,
        ) = dgl.sampling.global_uniform_negative_sampling(
            g,
            num_samples=num_samples,
            exclude_self_loops=True,
            replace=False,
        )

        negative_source_nodes = (
            negative_source_nodes.to(
                device=device
            )
        )
        negative_target_nodes = (
            negative_target_nodes.to(
                device=device
            )
        )

        return (
            negative_source_nodes,
            negative_target_nodes,
        )

    def edge_reconstruction_loss(
        self,
        g,
        node_embeddings,
        masked_edge_indices,
    ):

        if masked_edge_indices.numel() == 0:
            return node_embeddings.sum() * 0.0

        device = node_embeddings.device

        all_source_nodes, all_target_nodes = (
            g.edges()
        )
        all_source_nodes = all_source_nodes.to(
            device=device
        )
        all_target_nodes = all_target_nodes.to(
            device=device
        )
        masked_edge_indices = (
            masked_edge_indices.to(
                device=device
            )
        )

        positive_source_nodes = (
            all_source_nodes[
                masked_edge_indices
            ]
        )
        positive_target_nodes = (
            all_target_nodes[
                masked_edge_indices
            ]
        )

        (
            negative_source_nodes,
            negative_target_nodes,
        ) = self.sample_structure_negative_edges(
            g=g,
            num_samples=(
                positive_source_nodes.numel()
            ),
            device=device,
        )

        positive_logits = (
            self.score_reconstructed_edges(
                node_embeddings=node_embeddings,
                source_nodes=positive_source_nodes,
                target_nodes=positive_target_nodes,
            )
        )

        negative_logits = (
            self.score_reconstructed_edges(
                node_embeddings=node_embeddings,
                source_nodes=negative_source_nodes,
                target_nodes=negative_target_nodes,
            )
        )

        reconstruction_logits = torch.cat(
            [
                positive_logits,
                negative_logits,
            ],
            dim=0,
        )

        reconstruction_targets = torch.cat(
            [
                torch.ones_like(
                    positive_logits
                ),
                torch.zeros_like(
                    negative_logits
                ),
            ],
            dim=0,
        )

        edge_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                reconstruction_logits,
                reconstruction_targets,
            )
        )

        return edge_loss

    def forward(self,g,x,drop_g1=None, drop_g2=None,epoch=None,total_epochs=None,):
        if self.adrm_mode == "adaptive_dual_recon":
            return self.forward_adaptive_dual_recon(g=g,x=x,drop_g1=drop_g1,drop_g2=drop_g2,epoch=epoch,total_epochs=total_epochs,)
        # mask
        pre_use_g, mask_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self.mask_rate)
        # use_g = drop_g1 if drop_g1 is not None else g
        # # multi-layer encoder
        # Encode = []
        # for i, layer in enumerate(self.encoder):
        #     enc_rep = layer(use_g, mask_x)
        #     Encode.append(enc_rep)

        use_g = drop_g1 if drop_g1 is not None else g
        encoder_edge_weight = (use_g.edata["edge_weight"] if "edge_weight" in use_g.edata else None)

        # multi-layer encoder
        Encode = []
        for i, layer in enumerate(self.encoder):
            enc_rep = layer(use_g, mask_x,edge_weight=encoder_edge_weight)
            Encode.append(enc_rep)   
            

        Es = torch.stack(Encode, dim=1)  # (N, M, D * K)
        Es = torch.mean(Es, dim=1)
        # encode_to_decode
        origin_rep = self.encoder_to_decoder(Es)
        # decode
        Decode = []
        for i in range(self.num_remasking):
            # remask
            rep = origin_rep.clone()
            rep, remask_nodes, rekeep_nodes = self.random_remask(pre_use_g, rep, self.remask_rate)

            decoder_edge_weight = (pre_use_g.edata["edge_weight"] if "edge_weight" in pre_use_g.edata else None)
            # multi-layer decoder
            for i, layer in enumerate(self.decoder):
                recon = layer(pre_use_g, rep, edge_weight=decoder_edge_weight).flatten(1)
                Decode.append(recon)

            Ds = torch.stack(Decode, dim=1)  # (N, M, D * K)
            Ds = torch.mean(Ds, dim=1)

        return  self.decoder_to_contrastive(Ds)

    def forward_adaptive_dual_recon(
        self,
        g,
        x,
        drop_g1=None,
        drop_g2=None,
        epoch=None,
        total_epochs=None,
    ):

        (
            masked_x,
            target_x,
            mask_nodes,
            _keep_nodes,
            _current_mask_rate,
        ) = self.dynamic_attribute_mask(
            g=g,
            x=x,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        self.last_attr_mask_rate = float(_current_mask_rate)

        use_g = (
            drop_g1
            if drop_g1 is not None
            else g
        )
        (
            encoder_edge_weight,
            masked_edge_indices,
            _kept_edge_indices,
            current_edge_mask_rate,
            _edge_mask_probabilities,
        ) = self.adaptive_metapath_edge_mask(
            g=use_g,
            x=x,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        self.last_masked_edge_count = int(
            masked_edge_indices.numel()
        )
        self.last_edge_mask_rate = float(
            current_edge_mask_rate
        )

        encode_results = []

        for layer in self.encoder:
            encoded_representation = layer(
                use_g,
                masked_x,
                edge_weight=encoder_edge_weight,
            )
            encode_results.append(
                encoded_representation
            )

        encoder_embedding = torch.stack(
            encode_results,
            dim=1,
        )
        encoder_embedding = torch.mean(
            encoder_embedding,
            dim=1,
        )

        decoder_input = self.encoder_to_decoder(
            encoder_embedding
        )

        decode_results = []
        decoder_edge_weight = (
            g.edata["edge_weight"]
            if "edge_weight" in g.edata
            else None
        )

        for _ in range(self.num_remasking):
            remasked_representation = (
                decoder_input.clone()
            )
            (
                remasked_representation,
                _remask_nodes,
                _rekeep_nodes,
            ) = self.random_remask(
                g,
                remasked_representation,
                self.remask_rate,
            )

            for layer in self.decoder:
                reconstructed_representation = layer(
                    g,
                    remasked_representation,
                    edge_weight=decoder_edge_weight,
                ).flatten(1)

                decode_results.append(
                    reconstructed_representation
                )

        decoder_embedding = torch.stack(
            decode_results,
            dim=1,
        )
        decoder_embedding = torch.mean(
            decoder_embedding,
            dim=1,
        )
        self.last_attr_loss = (
            self.attribute_reconstruction_loss(
                prediction=decoder_embedding,
                target=target_x,
                mask_nodes=mask_nodes,
            )
        )

        decoder_embedding = (
            self.decoder_to_contrastive(
                decoder_embedding
            )
        )
        if (
            encoder_embedding.shape
            != decoder_embedding.shape
        ):
            raise RuntimeError(
                "H and E must have the same "
                "shape before computing S=H+E, "
                "but got "
                f"H={tuple(encoder_embedding.shape)} "
                "and "
                f"E={tuple(decoder_embedding.shape)}."
            )

        self.last_encoder_embedding = (
            encoder_embedding
        )
        self.last_intent_embedding = (
            decoder_embedding
        )
        self.last_bottleneck_embedding = (
            encoder_embedding
            + decoder_embedding
        )
        
        self.last_edge_loss = (
            self.edge_reconstruction_loss(
                g=use_g,
                node_embeddings=decoder_embedding,
                masked_edge_indices=(
                    masked_edge_indices
                ),
            )
        )
        return decoder_embedding



    def deterministic_forward(self, g, x):
        """测试阶段关闭随机mask和remask。"""
        edge_weight = (g.edata["edge_weight"] if "edge_weight" in g.edata else None)


        encode_results = []

        for layer in self.encoder:
            encoded_representation = layer(g,x,edge_weight=edge_weight)
            encode_results.append(encoded_representation)

        encoded = torch.stack(encode_results,dim=1)
        encoded = torch.mean(encoded, dim=1)

        decoder_input = self.encoder_to_decoder(encoded)


        decode_results = []

        for layer in self.decoder:
            reconstructed = layer(g,decoder_input,edge_weight=edge_weight).flatten(1)
            decode_results.append(reconstructed)

        decoded = torch.stack(decode_results,dim=1)
        decoded = torch.mean(decoded, dim=1)

        return self.decoder_to_contrastive(decoded)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):

        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes
