import argparse
import ast


REPORTING_SEEDS = (2024, 42, 3407, 2025, 2026)
FROZEN_SPLIT_SEED = 2024
PAPER_EVALUATION_CUTOFFS = (5, 10, 20)
PAPER_DATA_ROOT = "D:/AMDRRec/data/process"

def parse_args():
    parser = argparse.ArgumentParser(description="AMDRRec_main")

    parser.add_argument(
        "--seed",
        type=int,
        default=REPORTING_SEEDS[0],
        help=(
            "Model-training seed. Formal revision runs use exactly: "
            "2024, 42, 3407, 2025, and 2026."
        ),
    )
    parser.add_argument("--dataset", default="movielens-1m", help="Dataset to use, default: Movielens")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning Rate")
    parser.add_argument('--hdcm_fv_weight', '--ssl_lambda', dest='hdcm_fv_weight', type=float, default=0.02, help='Weight of the HDCM fused-view SCCF objective.')
    parser.add_argument('--hdcm_ii_weight', '--intra_lambda', dest='hdcm_ii_weight', type=float, default=0.001, help='Weight of the HDCM intent-intent InfoNCE objective.')
    parser.add_argument('--vib_beta', '--ib_lambda', dest='vib_beta', type=float, default=0.001, help='Compression coefficient of the variational information bottleneck.')
    parser.add_argument('--lambda_IBR', '--lambda_ibr', dest='lambda_IBR', type=float, default=0.01, help='Recommendation-preservation coefficient of the path-wise VIB BPR objective.')
    parser.add_argument(
        '--vib_latent_dim',
        '--vib_latent_size',
        '--IB_size',
        dest='vib_latent_dim',
        type=int,
        default=4,
        help=(
            'Latent dimension d_z of Z in the path-wise VIB. '
            'Each compressor emits 2*d_z values for mu and sigma.'
        ),
    )
    parser.add_argument("--reg_lambda", type=float, default=0.0001, help="Regularizations")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature")
    parser.add_argument("--sparsity_test", type=int, default=0, help="sparsity_test")
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--top_K', type=str, default="[5, 10, 20]", help='Evaluation cutoffs K; the paper reports 5, 10, and 20.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=200, help='batch size')
    parser.add_argument("--verbose", type=int, default=1, help="Test interval")
    parser.add_argument('--GCN_layer', type=int, default=2, help="the layer number of GCN")
    parser.add_argument(
        "--data_path",
        nargs="?",
        default=PAPER_DATA_ROOT,
        help=(
            "Frozen constrained per-user 8:1:1 data root. Interaction graphs "
            "and interaction-derived meta-path graphs must be built from its "
            "training split only."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    # hete_hyper
    parser.add_argument("--in_size", default=64, type=int, help="Initial dimension size for entities.")
    parser.add_argument("--out_size", default=64, type=int, help="Output dimension size for entities.")
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--remask_rate', type=float, default=0.1)
    parser.add_argument('--num_remasking', type=int, default=1)
    parser.add_argument("--adrm_mode", "--bae_mode", dest="adrm_mode", type=str, choices=["v1", "adaptive_dual_recon"],default="adaptive_dual_recon",help=("ADRM implementation mode; adaptive_dual_recon is the AMDRRec_main default."),)
    parser.add_argument("--attr_mask_rate_start",type=float,default=0.1,help=("Initial node attribute mask rate for ADRM."),)
    parser.add_argument("--attr_mask_rate_end",type=float,default=0.3,help=("Final node attribute mask rate for ADRM."),)
    parser.add_argument("--attr_sce_alpha",type=float,default=2.0,help=("Exponent alpha of the SCE attribute "   "reconstruction loss."),)
    parser.add_argument("--lambda_attr", "--attr_recon_weight", dest="lambda_attr", type=float,default=1.0,help=("Coefficient lambda_attr of the ADRM attribute-reconstruction objective."),)
    parser.add_argument("--edge_mask_rate_start",type=float,default=0.1,help=("Initial meta-path edge mask rate inside ADRM."),)
    parser.add_argument("--edge_mask_rate_end",type=float,default=0.3,help=("Final meta-path edge mask rate inside ADRM."),)
    parser.add_argument("--lambda_edge", "--edge_recon_weight", dest="lambda_edge", type=float,default=1.0,help=("Coefficient lambda_edge of the ADRM edge-structure reconstruction objective." ),)

    parser.add_argument("--edge_threshold",type=float,default=0.1,help="Threshold for the straight-through edge gate.")    
    #parser.add_argument("--router_mode",type=str,choices=["uniform"],default="uniform",help="Meta-path routing mode.",)
    parser.add_argument("--router_mode",type=str,choices=["uniform","global",],default="uniform",help="Meta-path routing mode.",) 
    parser.add_argument("--amr_mode", "--pre_bae_router_mode", dest="amr_mode", type=str,choices=["none", "uniform", "global"],default="global",help="Adaptive meta-path routing mode applied before ADRM.",)
    parser.add_argument("--residual_fusion_gamma", "--fusion_scale", dest="residual_fusion_gamma", type=float,default=1e-5,help="Residual fusion coefficient gamma in Eq. (27).",)
    parser.add_argument("--router_residual_scale",type=float,default=0.1,help=("Scale applied only to the zero-centered " "Router residual."),)
    parser.add_argument("--router_hidden_dim",type=int,default=64,help="Hidden dimension of the learnable meta-path Router.",)    
    parser.add_argument("--run_name",type=str,default="AMDRRec_main",help="Experiment name used for saving results.")
    parser.add_argument(
        "--run_stage",
        type=str,
        choices=["engineering", "train_validate", "sealed_test"],
        default="engineering",
        help=(
            "engineering reads training data only; train_validate reads train and "
            "validation but is forbidden from reading test; sealed_test loads a frozen "
            "validation-selected checkpoint and evaluates test exactly once."
        ),
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=1,
        help="Evaluate validation metrics every N epochs in train_validate stage.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Number of validation checks without improvement before stopping.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-5,
        help="Minimum NDCG@20 increase required to reset early stopping.",
    )
    parser.add_argument(
        "--frozen_checkpoint",
        type=str,
        default="",
        help="Validation-selected checkpoint required by sealed_test stage.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["engineering", "formal"],
        default=None,
        help="Deprecated compatibility option. Use --run_stage for revision experiments.",
    )



    args = parser.parse_args()

    args.ssl_lambda = args.hdcm_fv_weight
    args.intra_lambda = args.hdcm_ii_weight
    args.ib_lambda = args.vib_beta
    args.lambda_ibr = args.lambda_IBR
    args.vib_latent_size = args.vib_latent_dim
    args.IB_size = args.vib_latent_dim
    args.bae_mode = args.adrm_mode
    args.pre_bae_router_mode = args.amr_mode
    args.attr_recon_weight = args.lambda_attr
    args.edge_recon_weight = args.lambda_edge
    args.fusion_scale = args.residual_fusion_gamma
    if args.run_mode is not None:
        if args.run_stage != "engineering":
            parser.error("Do not combine deprecated --run_mode with --run_stage.")
        args.run_stage = (
            "sealed_test" if args.run_mode == "formal" else "engineering"
        )
    if args.validation_interval < 1:
        parser.error("--validation_interval must be at least 1.")
    if args.early_stop_patience < 1:
        parser.error("--early_stop_patience must be at least 1.")
    if args.early_stop_min_delta < 0:
        parser.error("--early_stop_min_delta must be non-negative.")
    if args.run_stage == "sealed_test" and not args.frozen_checkpoint:
        parser.error("--sealed_test requires --frozen_checkpoint.")
    if args.run_stage in {"train_validate", "sealed_test"} and args.seed not in REPORTING_SEEDS:
        parser.error(
            "Revision experiments require one of the five frozen reporting seeds: "
            "2024, 42, 3407, 2025, or 2026."
        )
    if args.run_stage in {"train_validate", "sealed_test"} and args.router_mode != "uniform":
        parser.error(
            "Paper-aligned runs require --router_mode uniform because Equation "
            "(27) uses a fixed residual sum; adaptive path selection is controlled "
            "separately by --amr_mode global."
        )
    if args.vib_latent_dim < 1:
        parser.error("--vib_latent_dim must be at least 1.")
    if args.lambda_attr < 0 or args.lambda_edge < 0:
        parser.error("--lambda_attr and --lambda_edge must be non-negative.")
    if args.residual_fusion_gamma < 0:
        parser.error("--residual_fusion_gamma must be non-negative.")
    for name in (
        "hdcm_fv_weight",
        "hdcm_ii_weight",
        "vib_beta",
        "lambda_IBR",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative.")
    for start_name, end_name in (
        ("attr_mask_rate_start", "attr_mask_rate_end"),
        ("edge_mask_rate_start", "edge_mask_rate_end"),
    ):
        start = getattr(args, start_name)
        end = getattr(args, end_name)
        if not 0.0 <= start <= end <= 1.0:
            parser.error(
                f"--{start_name} and --{end_name} must satisfy "
                "0 <= start <= end <= 1."
            )
    try:
        evaluation_cutoffs = tuple(int(k) for k in ast.literal_eval(args.top_K))
    except (SyntaxError, ValueError, TypeError) as error:
        parser.error(f"--top_K must be a Python list of integers: {error}")
    if any(k < 1 for k in evaluation_cutoffs):
        parser.error("Every --top_K cutoff must be positive.")
    if args.run_stage in {"train_validate", "sealed_test"} and set(evaluation_cutoffs) != set(PAPER_EVALUATION_CUTOFFS):
        parser.error("Formal revision runs require --top_K '[5, 10, 20]'.")
    return args
