import utility.losses
import utility.tools
from .ViewLearner import ViewLearner
from .adrm import AdaptiveDualReconstructionModule
from .amr import AdaptiveMetaPathRouter, UniformMetaPathRouter
from .hdcm import HybridDualContrastiveModule
from .vib import VariationalInformationBottleneck
from .uniform_router import UniformMeanRouter
from .global_router import GlobalMetaPathRouter
import torch
from torch import nn

class AMDRRec(nn.Module):

    LOSS_COMPONENT_NAMES = (
        "bpr_loss",
        "reg_loss",
        "hdcm_fv_loss",
        "hdcm_ii_loss",
        "vib_loss",
        "attr_recon_loss",
        "edge_recon_loss",
    )

    def __init__(self, config, dataset, user_g, item_g, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config.reg_lambda)
        self.hdcm_fv_weight = float(
            getattr(
                self.config,
                "hdcm_fv_weight",
                getattr(self.config, "ssl_lambda", 0.02),
            )
        )
        self.vib_beta = float(
            getattr(
                self.config,
                "vib_beta",
                getattr(self.config, "ib_lambda", 0.001),
            )
        )
        self.lambda_IBR = float(
            getattr(
                self.config,
                "lambda_IBR",
                getattr(self.config, "lambda_ibr", 0.01),
            )
        )
        self.hdcm_ii_weight = float(
            getattr(
                self.config,
                "hdcm_ii_weight",
                getattr(self.config, "intra_lambda", 0.001),
            )
        )
        self.temperature = float(self.config.temperature)
        self.lambda_attr = float(
            getattr(
                self.config,
                "lambda_attr",
                getattr(self.config, "attr_recon_weight", 1.0),
            )
        )

        if self.lambda_attr < 0.0:
            raise ValueError(
                "lambda_attr must be "
                "non-negative, but got "
                f"{self.lambda_attr}."
            )        

        self.lambda_edge = float(
            getattr(
                self.config,
                "lambda_edge",
                getattr(self.config, "edge_recon_weight", 1.0),
            )
        )

        if self.lambda_edge < 0.0:
            raise ValueError(
                "lambda_edge must be "
                "non-negative, but got "
                f"{self.lambda_edge}."
            )
        #self.view_learner = ViewLearner(input_dim=self.config.dim, output_dim=self.config.dim)
        self.view_learner = ViewLearner(input_dim=self.config.dim, output_dim=self.config.dim, edge_threshold=self.config.edge_threshold)

        self.vib_latent_dim = int(
            getattr(
                self.config,
                "vib_latent_dim",
                getattr(self.config, "vib_latent_size", 4),
            )
        )
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=int(self.config.dim))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=int(self.config.dim))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix
        self.Graph = utility.tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy
        self.activation = nn.Sigmoid()
        # hete_information
        self.uu_graph = user_g
        self.ii_graph = item_g

         self.num_adrm_branches = 2

        if len(self.uu_graph) < self.num_adrm_branches:
            raise ValueError(
                "At least two user-side candidate graphs are required."
            )

        if len(self.ii_graph) < self.num_adrm_branches:
            raise ValueError(
                "At least two item-side candidate graphs are required."
            )


        self.router_mode = str(getattr(self.config,"router_mode","uniform",)).lower()
        self.amr_mode = str(
            getattr(
                self.config,
                "amr_mode",
                getattr(self.config, "pre_bae_router_mode", "none"),
            )
        ).lower()
        self.residual_fusion_gamma = float(
            getattr(
                self.config,
                "residual_fusion_gamma",
                getattr(self.config, "fusion_scale", 1e-5),
            )
        )
        self.router_residual_scale = float(getattr(self.config,"router_residual_scale", 0.1,))       
        self.router_hidden_dim = int(getattr(self.config,"router_hidden_dim",64,))
        self.last_user_amr_route = None
        self.last_item_amr_route = None
    
        self.user_adrm_branches = nn.ModuleList()
        self.vib = VariationalInformationBottleneck(
            input_dim=self.config.dim,
            latent_dim=self.vib_latent_dim,
            beta=self.vib_beta,
            recommendation_weight=self.lambda_IBR,
        )
        self.hdcm = HybridDualContrastiveModule(
            intent_intent_weight=self.hdcm_ii_weight,
            fused_view_weight=self.hdcm_fv_weight,
        )
        for i in range(self.num_adrm_branches):
            self.user_adrm_branches.append(AdaptiveDualReconstructionModule(in_dim=config.in_size,hidden_dim=config.out_size,enc_num_layer=config.enc_num_layer,
                    dec_num_layer=config.dec_num_layer,mask_rate=config.mask_rate,remask_rate=config.remask_rate,num_remasking=config.num_remasking,
                    adrm_mode=getattr(config,"adrm_mode",getattr(config,"bae_mode","v1"),),attr_mask_rate_start=getattr(config,"attr_mask_rate_start",0.1,),
                    attr_mask_rate_end=getattr(config,"attr_mask_rate_end",0.3,),                    
                    attr_sce_alpha=getattr(config,"attr_sce_alpha",2.0,),
                    edge_mask_rate_start=getattr(config,"edge_mask_rate_start",0.1,),
                    edge_mask_rate_end=getattr(config,"edge_mask_rate_end",0.3,),))
            
        self.item_adrm_branches = nn.ModuleList()

        for i in range(self.num_adrm_branches):
            self.item_adrm_branches.append(AdaptiveDualReconstructionModule(in_dim=config.in_size,hidden_dim=config.out_size,enc_num_layer=config.enc_num_layer,
                    dec_num_layer=config.dec_num_layer,mask_rate=config.mask_rate,remask_rate=config.remask_rate,num_remasking=config.num_remasking,
                    adrm_mode=getattr(config,"adrm_mode",getattr(config,"bae_mode","v1"),),
                    attr_mask_rate_start=getattr(config,"attr_mask_rate_start",0.1,),
                    attr_mask_rate_end=getattr(config,"attr_mask_rate_end",0.3,),
                    attr_sce_alpha=getattr(config,"attr_sce_alpha",2.0,),
                    edge_mask_rate_start=getattr(config,"edge_mask_rate_start",0.1,),
                    edge_mask_rate_end=getattr(config,"edge_mask_rate_end",0.3,),))

        if self.router_mode == "uniform":
            self.user_router = UniformMeanRouter(
                residual_fusion_gamma=self.residual_fusion_gamma,
            )
            self.item_router = UniformMeanRouter(
                residual_fusion_gamma=self.residual_fusion_gamma,
            )

        elif self.router_mode == "global":
            self.user_router = GlobalMetaPathRouter(embedding_dim=int(self.config.dim),hidden_dim=self.router_hidden_dim, fusion_scale=self.residual_fusion_gamma,
                router_residual_scale=(self.router_residual_scale),)

            self.item_router = GlobalMetaPathRouter(embedding_dim=int(self.config.dim),hidden_dim=self.router_hidden_dim,fusion_scale=self.residual_fusion_gamma,
                router_residual_scale=(self.router_residual_scale),)

        else:
            raise ValueError(f"Unsupported router mode: "  f"{self.router_mode}")

        if self.amr_mode == "none":
            self.user_amr = None
            self.item_amr = None

        elif self.amr_mode == "uniform":
            self.user_amr = UniformMetaPathRouter()
            self.item_amr = UniformMetaPathRouter()

        elif self.amr_mode == "global":
            self.user_amr = AdaptiveMetaPathRouter(num_paths=len(self.uu_graph),top_k=self.num_adrm_branches,)
            self.item_amr = AdaptiveMetaPathRouter(num_paths=len(self.ii_graph),top_k=self.num_adrm_branches,)
        else:
            raise ValueError("Unsupported AMR mode: " f"{self.amr_mode}")

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []

        for layer in range(int(self.config.GCN_layer)):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def build_path_embeddings(self, base_embeddings, heterogeneous_embeddings):
        if len(heterogeneous_embeddings) == 0:
            raise ValueError("At least one heterogeneous path is required.")

        path_embeddings = [
            base_embeddings + heterogeneous_embedding
            for heterogeneous_embedding in heterogeneous_embeddings
        ]
        return torch.stack(path_embeddings, dim=1)

    def fuse_path_embeddings(self, base_embeddings, path_embeddings):
        """Implement Equation (27): E_bar = E' + gamma * sum_m F^m."""
        fused_embeddings = base_embeddings

        for path_index in range(path_embeddings.size(1)):
            fused_embeddings = (
                fused_embeddings
                + self.residual_fusion_gamma
                * path_embeddings[:, path_index, :]
            )

        return fused_embeddings

    def get_k_path_info_nce_loss(self,path_embeddings,node_indices,):
        if path_embeddings.dim() != 3:
            raise ValueError("path_embeddings must have shape "  "[num_nodes, num_paths, embedding_dim].")

        num_paths = path_embeddings.size(1)

        if num_paths < 2:
            raise ValueError("At least two paths are required for InfoNCE.")

        pair_losses = []

        for first_index in range(num_paths):
            for second_index in range(first_index + 1,num_paths,):
                pair_losses.append(
                    utility.losses.get_InfoNCE_loss(path_embeddings[node_indices,first_index,:],
                        path_embeddings[node_indices,second_index,:],self.temperature,))

        if len(pair_losses) == 1:
            return pair_losses[0]

        return torch.mean(torch.stack(pair_losses))

    def get_k_path_sccf_loss(
        self,
        path_embeddings,
        node_indices,
    ):
        if path_embeddings.dim() != 3:
            raise ValueError(
                "path_embeddings must have shape "
                "[num_nodes, num_paths, embedding_dim]."
            )

        num_paths = path_embeddings.size(1)

        if num_paths < 2:
            raise ValueError(
                "At least two paths are required for SCCF."
            )

        pair_losses = []

        for first_index in range(num_paths):
            for second_index in range(
                first_index + 1,
                num_paths,
            ):
                pair_losses.append(
                    utility.losses.get_cross_view_sccf_loss(
                        first_view_embeddings=(
                            path_embeddings[
                                :,
                                first_index,
                                :,
                            ]
                        ),
                        second_view_embeddings=(
                            path_embeddings[
                                :,
                                second_index,
                                :,
                            ]
                        ),
                        node_indices=node_indices,
                        temperature=self.temperature,
                    )
                )

        if len(pair_losses) == 1:
            return pair_losses[0]

        return torch.mean(
            torch.stack(pair_losses)
        )

    def forward(self, user, positive, negative, epoch=None):
        user_embeddings, item_embeddings = self.aggregate()

        user_amr_weights = None
        item_amr_weights = None

        user_hard_topk_route = None
        item_hard_topk_route = None

        user_selected_indices = torch.arange(
            self.num_adrm_branches,
            dtype=torch.long,
            device=self.user_embedding.weight.device,
        )
        item_selected_indices = torch.arange(
            self.num_adrm_branches,
            dtype=torch.long,
            device=self.item_embedding.weight.device,
        )

        user_selected_gate_weights = torch.ones(
            self.num_adrm_branches,
            dtype=self.user_embedding.weight.dtype,
            device=self.user_embedding.weight.device,
        )
        item_selected_gate_weights = torch.ones(
            self.num_adrm_branches,
            dtype=self.item_embedding.weight.dtype,
            device=self.item_embedding.weight.device,
        )

        if isinstance(
            self.user_amr,
            AdaptiveMetaPathRouter,
        ):
            user_hard_topk_route = (
                self.user_amr.get_hard_topk_route(
                    node_embeddings=self.user_embedding.weight,
                    num_paths=len(self.uu_graph),
                )
            )
            user_selected_indices = (
                user_hard_topk_route["topk_indices"]
            )
            user_selected_gate_weights = (
                user_hard_topk_route["selected_gate_weights"]
            )
        elif self.user_amr is not None:
            user_amr_weights = (
                self.user_amr(
                    node_embeddings=self.user_embedding.weight,
                    num_paths=len(self.uu_graph),
                )
            )

        if isinstance(
            self.item_amr,
            AdaptiveMetaPathRouter,
        ):
            item_hard_topk_route = (
                self.item_amr.get_hard_topk_route(
                    node_embeddings=self.item_embedding.weight,
                    num_paths=len(self.ii_graph),
                )
            )
            item_selected_indices = (
                item_hard_topk_route["topk_indices"]
            )
            item_selected_gate_weights = (
                item_hard_topk_route["selected_gate_weights"]
            )
        elif self.item_amr is not None:
            item_amr_weights = (
                self.item_amr(
                    node_embeddings=self.item_embedding.weight,
                    num_paths=len(self.ii_graph),
                )
            )

        self.last_user_amr_route = user_hard_topk_route
        self.last_item_amr_route = item_hard_topk_route


        # hete_emb
        hete_user_embedding = []
        hete_item_embedding = []
        hete_user_bottleneck_embedding = []
        hete_item_bottleneck_embedding = []
        attr_recon_losses = []
        attr_recon_valid_counts = []
        edge_recon_losses = []

        for branch_index, graph_index_tensor in enumerate(user_selected_indices):
            graph_index = int(graph_index_tensor.item())

            self.uu_graph[graph_index] = self.view_learner(self.uu_graph[graph_index],self.user_embedding.weight,)
            user_bae_graph = self.uu_graph[graph_index]

             if user_amr_weights is not None:
                user_bae_graph = (
                    self.user_amr.route_graph(
                        graph=user_bae_graph,
                        node_path_weights=(
                            user_amr_weights[
                                :,
                                graph_index,
                            ]
                        ),
                        num_paths=len(self.uu_graph),
                    )
                )

            user_intent_embedding = self.user_adrm_branches[
                branch_index
            ](
                user_bae_graph,
                self.user_embedding.weight,
                epoch=epoch,
                total_epochs=int(
                    self.config.epochs
                ),
            ).flatten(1)
            current_attr_loss = (
                self.user_adrm_branches[
                    branch_index
                ].last_attr_loss
            )

            if current_attr_loss is not None:
                attr_recon_losses.append(
                    current_attr_loss
                )
                attr_recon_valid_counts.append(
                    self.user_adrm_branches[
                        branch_index
                    ].last_attr_valid_count
                )
            current_edge_loss = (
                self.user_adrm_branches[
                    branch_index
                ].last_edge_loss
            )

            if current_edge_loss is not None:
                edge_recon_losses.append(
                    current_edge_loss
                )
            user_intent_embedding = (
                user_selected_gate_weights[branch_index]
                * user_intent_embedding
            )

            hete_user_embedding.append(
                user_intent_embedding)

            user_bottleneck_embedding = (
                self.user_adrm_branches[
                    branch_index
                ].last_bottleneck_embedding
            )

            if user_bottleneck_embedding is None:
                user_bottleneck_embedding = (
                    user_intent_embedding
                )
            else:
                user_bottleneck_embedding = (
                    user_selected_gate_weights[
                        branch_index
                    ]
                    * user_bottleneck_embedding
                )

            hete_user_bottleneck_embedding.append(
                user_bottleneck_embedding
            )
       
        for branch_index, graph_index_tensor in enumerate(
            item_selected_indices
        ):
            graph_index = int(
                graph_index_tensor.item()
            )

            self.ii_graph[graph_index] = self.view_learner(
                self.ii_graph[graph_index],
                self.item_embedding.weight,
            )
            item_bae_graph = self.ii_graph[graph_index]

            if item_amr_weights is not None:
                item_bae_graph = (
                    self.item_amr.route_graph(
                        graph=item_bae_graph,
                        node_path_weights=(
                            item_amr_weights[
                                :,
                                graph_index,
                            ]
                        ),
                        num_paths=len(self.ii_graph),
                    )
                )

            item_intent_embedding = self.item_adrm_branches[
                branch_index
            ](
                item_bae_graph,
                self.item_embedding.weight,
                epoch=epoch,
                total_epochs=int(
                    self.config.epochs
                ),
            ).flatten(1)
            current_attr_loss = (
                self.item_adrm_branches[
                    branch_index
                ].last_attr_loss
            )

            if current_attr_loss is not None:
                attr_recon_losses.append(
                    current_attr_loss
                )
                attr_recon_valid_counts.append(
                    self.item_adrm_branches[
                        branch_index
                    ].last_attr_valid_count
                )
            current_edge_loss = (
                self.item_adrm_branches[
                    branch_index
                ].last_edge_loss
            )

            if current_edge_loss is not None:
                edge_recon_losses.append(
                    current_edge_loss
                )
            item_intent_embedding = (
                item_selected_gate_weights[branch_index]
                * item_intent_embedding
            )

            hete_item_embedding.append(
                item_intent_embedding
            )

            item_bottleneck_embedding = (
                self.item_adrm_branches[
                    branch_index
                ].last_bottleneck_embedding
            )

            if item_bottleneck_embedding is None:
                item_bottleneck_embedding = (
                    item_intent_embedding
                )
            else:
                item_bottleneck_embedding = (
                    item_selected_gate_weights[
                        branch_index
                    ]
                    * item_bottleneck_embedding
                )

            hete_item_bottleneck_embedding.append(
                item_bottleneck_embedding
            )


            hete_user_bottleneck_embedding,
            dim=1,
        )
        item_vib_representations = torch.stack(
            hete_item_bottleneck_embedding,
            dim=1,
        )
        vib_loss = self.vib(
            user_path_representations=user_vib_representations,
            item_path_representations=item_vib_representations,
            user_indices=user,
            positive_item_indices=positive,
            negative_item_indices=negative,
        )
        user_path_embeddings = self.build_path_embeddings(user_embeddings,hete_user_embedding,)
        item_path_embeddings = self.build_path_embeddings(item_embeddings,hete_item_embedding,)

        all_user_embeddings, user_router_weights = self.user_router(user_embeddings,user_path_embeddings,)
        all_item_embeddings, item_router_weights = self.item_router(item_embeddings,item_path_embeddings,)

        heterogeneous_user_path_embeddings = torch.stack(hete_user_embedding,dim=1,)
        heterogeneous_item_path_embeddings = torch.stack(hete_item_embedding,dim=1,)

        user_intent_intent_loss = self.get_k_path_info_nce_loss(heterogeneous_user_path_embeddings,user.long(),)
        item_intent_intent_loss = self.get_k_path_info_nce_loss(heterogeneous_item_path_embeddings,positive.long(),)

        hdcm_intent_loss = self.hdcm.intent_intent_loss(
            user_intent_intent_loss,
            item_intent_intent_loss,
        )

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        # bpr_loss
        bpr_loss = utility.losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        # reg_loss
        reg_loss = utility.losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        user_fused_view_loss = self.get_k_path_sccf_loss(
            user_path_embeddings,
            user.long(),
        )

        item_fused_view_loss = self.get_k_path_sccf_loss(
            item_path_embeddings,
            positive.long(),
        )

        hdcm_fused_view_loss = self.hdcm.fused_view_loss(
            user_fused_view_loss,
            item_fused_view_loss,
        )

        loss_list = [
            bpr_loss,
            reg_loss,
            hdcm_fused_view_loss,
            hdcm_intent_loss,
            vib_loss,
        ]

        if len(attr_recon_losses) > 0:
            expected_attr_loss_count = (
                2 * self.num_adrm_branches
            )

            if (
                len(attr_recon_losses)
                != expected_attr_loss_count
            ):
                raise RuntimeError(
                    "Expected "
                    f"{expected_attr_loss_count} "
                    "attribute reconstruction "
                    "losses, but got "
                    f"{len(attr_recon_losses)}."
                )

            attr_loss_stack = torch.stack(
                attr_recon_losses
            )
            if len(attr_recon_valid_counts) != expected_attr_loss_count:
                raise RuntimeError(
                    "Attribute reconstruction loss/count mismatch: "
                    f"{len(attr_recon_losses)} losses and "
                    f"{len(attr_recon_valid_counts)} counts."
                )

            attr_count_tensor = attr_loss_stack.new_tensor(
                attr_recon_valid_counts
            )
            total_attr_count = attr_count_tensor.sum()
            if total_attr_count.item() > 0:
                raw_attr_recon_loss = (
                    attr_loss_stack * attr_count_tensor
                ).sum() / total_attr_count
            else:
                raw_attr_recon_loss = attr_loss_stack.sum() * 0.0

            attr_recon_loss = self.lambda_attr * raw_attr_recon_loss

            loss_list.append(
                attr_recon_loss
            )

        if len(edge_recon_losses) > 0:
            expected_edge_loss_count = (
                2 * self.num_adrm_branches
            )

            if (
                len(edge_recon_losses)
                != expected_edge_loss_count
            ):
                raise RuntimeError(
                    "Expected "
                    f"{expected_edge_loss_count} "
                    "edge reconstruction "
                    "losses, but got "
                    f"{len(edge_recon_losses)}."
                )


            edge_recon_loss = self.lambda_edge * torch.mean(
                torch.stack(edge_recon_losses)
            )

            loss_list.append(
                edge_recon_loss
            )
        if len(loss_list) != len(self.LOSS_COMPONENT_NAMES):
            raise RuntimeError(
                "The complete AMDRRec objective must expose exactly "
                f"{len(self.LOSS_COMPONENT_NAMES)} loss components, "
                f"but got {len(loss_list)}."
            )

        self.last_loss_components = dict(
            zip(self.LOSS_COMPONENT_NAMES, loss_list)
        )
        return loss_list

    def get_deterministic_embeddings_for_test(self):
        user_embeddings, item_embeddings = self.aggregate()
        user_amr_weights = None
        item_amr_weights = None


        user_selected_indices = torch.arange(
            self.num_adrm_branches,
            dtype=torch.long,
            device=self.user_embedding.weight.device,
        )
        item_selected_indices = torch.arange(
            self.num_adrm_branches,
            dtype=torch.long,
            device=self.item_embedding.weight.device,
        )

        if isinstance(
            self.user_amr,
            AdaptiveMetaPathRouter,
        ):
            user_hard_topk_route = (
                self.user_amr.get_hard_topk_route(
                    node_embeddings=self.user_embedding.weight,
                    num_paths=len(self.uu_graph),
                )
            )
            user_selected_indices = (
                user_hard_topk_route["topk_indices"]
            )
        elif self.user_amr is not None:
            user_amr_weights = (
                self.user_amr(
                    node_embeddings=self.user_embedding.weight,
                    num_paths=len(self.uu_graph),
                )
            )

        if isinstance(
            self.item_amr,
            AdaptiveMetaPathRouter,
        ):
            item_hard_topk_route = (
                self.item_amr.get_hard_topk_route(
                    node_embeddings=self.item_embedding.weight,
                    num_paths=len(self.ii_graph),
                )
            )
            item_selected_indices = (
                item_hard_topk_route["topk_indices"]
            )
        elif self.item_amr is not None:
            item_amr_weights = (
                self.item_amr(
                    node_embeddings=self.item_embedding.weight,
                    num_paths=len(self.ii_graph),
                )
            )

        heterogeneous_user_embeddings = []
        heterogeneous_item_embeddings = []


        for branch_index, graph_index_tensor in enumerate(
            user_selected_indices
        ):
            graph_index = int(
                graph_index_tensor.item()
            )

            deterministic_graph = (
                self.view_learner.deterministic_forward(
                    self.uu_graph[graph_index],
                    self.user_embedding.weight,
                )
            )

            if user_amr_weights is not None:
                deterministic_graph = (
                    self.user_amr.route_graph(
                        graph=deterministic_graph,
                        node_path_weights=(
                            user_amr_weights[
                                :,
                                graph_index,
                            ]
                        ),
                        num_paths=len(self.uu_graph),
                    )
                )

            deterministic_embedding = (
                self.user_adrm_branches[
                    branch_index
                ].deterministic_forward(
                    deterministic_graph,
                    self.user_embedding.weight,
                )
            )

            heterogeneous_user_embeddings.append(
                deterministic_embedding.flatten(1)
            )

        for branch_index, graph_index_tensor in enumerate(
            item_selected_indices
        ):
            graph_index = int(
                graph_index_tensor.item()
            )

            deterministic_graph = (
                self.view_learner.deterministic_forward(
                    self.ii_graph[graph_index],
                    self.item_embedding.weight,
                )
            )

            if item_amr_weights is not None:
                deterministic_graph = (
                    self.item_amr.route_graph(
                        graph=deterministic_graph,
                        node_path_weights=(
                            item_amr_weights[
                                :,
                                graph_index,
                            ]
                        ),
                        num_paths=len(self.ii_graph),
                    )
                )

            deterministic_embedding = (
                self.item_adrm_branches[
                    branch_index
                ].deterministic_forward(
                    deterministic_graph,
                    self.item_embedding.weight,
                )
            )

            heterogeneous_item_embeddings.append(
                deterministic_embedding.flatten(1)
            )

        user_path_embeddings = self.build_path_embeddings(user_embeddings,heterogeneous_user_embeddings,)
        item_path_embeddings = self.build_path_embeddings(item_embeddings,heterogeneous_item_embeddings,)

        all_user_embeddings, user_router_weights = self.user_router(user_embeddings,user_path_embeddings,)
        all_item_embeddings, item_router_weights = self.item_router(item_embeddings,item_path_embeddings,)

        return all_user_embeddings, all_item_embeddings

    def get_rating_for_test(self,user,all_user_embeddings=None,all_item_embeddings=None):

        if (
            all_user_embeddings is None or all_item_embeddings is None
        ):
            (all_user_embeddings,all_item_embeddings) = self.get_deterministic_embeddings_for_test()

        user_embeddings = all_user_embeddings[user.long()]
        rating = self.activation(torch.matmul(user_embeddings,all_item_embeddings.t()))
        return rating


    def get_embedding(self):
        all_user_embeddings, all_item_embeddings = self.aggregate()
        return all_user_embeddings, all_item_embeddings


    @property
    def user_autoencoder(self):
        return self.user_adrm_branches

    @property
    def item_autoencoder(self):
        return self.item_adrm_branches

    @property
    def user_pre_bae_router(self):
        return self.user_amr

    @user_pre_bae_router.setter
    def user_pre_bae_router(self, value):
        self.user_amr = value

    @property
    def item_pre_bae_router(self):
        return self.item_amr

    @item_pre_bae_router.setter
    def item_pre_bae_router(self, value):
        self.item_amr = value

    @property
    def pre_bae_router_mode(self):
        return self.amr_mode

    @property
    def num_bae_branches(self):
        return self.num_adrm_branches

    @property
    def user_compressor(self):
        return self.vib.user_compressor

    @property
    def item_compressor(self):
        return self.vib.item_compressor

    @property
    def IB_size(self):
        return self.vib_latent_dim

    @property
    def IB_2_size(self):
        return self.vib_latent_dim


GCRec = AMDRRec
