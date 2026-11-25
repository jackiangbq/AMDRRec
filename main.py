import ast
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import time
import utility.batch_test
import utility.parser
import utility.losses
import utility.tools
import utility.trainer
from utility.data_loader import Data
from time import time
from model.amdrrec import AMDRRec
from utility.load_hete_data import get_meta_path_names, load_data
from utility.primary_anchored_gradient import (apply_primary_anchored_backward,)


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_amr_route_row(
    epoch,
    side,
    path_names,
    route,
    mean_path_logit_gradients,
    previous_top2,
    cumulative_switches,
):

    weights = route["global_path_weights"].detach().cpu().tolist()
    top2_indices = route["topk_indices"].detach().cpu().tolist()
    top2_names = [path_names[index] for index in top2_indices]
    selection_changed = int(
        previous_top2 is not None and top2_indices != previous_top2
    )
    cumulative_switches += selection_changed

    if mean_path_logit_gradients is None:
        gradient_values = None
        unselected_nonzero_gradient = None
    else:
        gradient_values = mean_path_logit_gradients.detach().cpu().tolist()
        selected = set(top2_indices)
        unselected_nonzero_gradient = all(
            abs(gradient_values[index]) > 0.0
            for index in range(len(gradient_values))
            if index not in selected
        )

    return (
        {
            "epoch": epoch,
            "side": side,
            "path_names": json.dumps(path_names, ensure_ascii=False),
            "path_weights": json.dumps(weights),
            "top2_indices": json.dumps(top2_indices),
            "top2_names": json.dumps(top2_names, ensure_ascii=False),
            "selection_changed": selection_changed,
            "cumulative_switches": cumulative_switches,
            "mean_path_logit_gradients": (
                "" if gradient_values is None else json.dumps(gradient_values)
            ),
            "unselected_nonzero_gradient": (
                "" if unselected_nonzero_gradient is None
                else int(unselected_nonzero_gradient)
            ),
        },
        top2_indices,
        cumulative_switches,
    )


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    args,
    result,
    selection_metric_name,
    selection_metric_value,
    protocol_identity,
):


    checkpoint = {
        "epoch": epoch,
        "dataset": args.dataset,
        "arguments": vars(args).copy(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": result,
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": selection_metric_value,
        "protocol_identity": protocol_identity,
    }

    torch.save(checkpoint, str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_protocol_identity(args):
    dataset_dir = (Path(args.data_path) / args.dataset).resolve()
    required = {
        "train": dataset_dir / "train.txt",
        "validation": dataset_dir / "validation.txt",
        "split_manifest": dataset_dir / "split_manifest.json",
        "graph_manifest": dataset_dir / "graph_manifest.json",
    }
    if args.run_stage == "sealed_test":
        required["test"] = dataset_dir / "test.txt"
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing revision protocol files: {missing}")

    split_manifest = json.loads(required["split_manifest"].read_text(encoding="utf-8"))
    if split_manifest.get("algorithm_version") != "amdrrec-user-aware-8-1-1-v1":
        raise ValueError("Unexpected split algorithm version.")
    if split_manifest.get("seed") != utility.parser.FROZEN_SPLIT_SEED:
        raise ValueError(
            "The frozen data split must use split seed "
            f"{utility.parser.FROZEN_SPLIT_SEED}."
        )
    output_hashes = split_manifest.get("output_files", {})
    for key, file_name in (("train", "train.txt"), ("validation", "validation.txt")):
        actual_hash = file_sha256(required[key])
        if output_hashes.get(file_name) != actual_hash:
            raise ValueError(f"Split hash mismatch for {file_name}.")
    if args.run_stage == "sealed_test":
        actual_test_hash = file_sha256(required["test"])
        if output_hashes.get("test.txt") != actual_test_hash:
            raise ValueError("Split hash mismatch for test.txt.")

    graph_manifest = json.loads(required["graph_manifest"].read_text(encoding="utf-8"))
    if graph_manifest.get("algorithm_version") != "amdrrec-training-only-graphs-v1":
        raise ValueError("Unexpected graph preparation algorithm version.")
    train_hash = file_sha256(required["train"])
    if graph_manifest.get("train_file", {}).get("sha256") != train_hash:
        raise ValueError("Graph manifest was not built from the active training split.")
    graph_hashes = {}
    for section in ("static_graphs", "interaction_derived_graphs"):
        for graph_name, metadata in graph_manifest.get(section, {}).items():
            graph_path = dataset_dir / graph_name
            if not graph_path.is_file():
                raise FileNotFoundError(f"Missing graph declared in manifest: {graph_path}")
            actual_hash = file_sha256(graph_path)
            if metadata.get("output_sha256") != actual_hash:
                raise ValueError(f"Graph hash mismatch: {graph_path}")
            graph_hashes[graph_name] = actual_hash

    return {
        "dataset_directory": str(dataset_dir),
        "split_seed": utility.parser.FROZEN_SPLIT_SEED,
        "split_manifest_sha256": file_sha256(required["split_manifest"]),
        "graph_manifest_sha256": file_sha256(required["graph_manifest"]),
        "train_sha256": train_hash,
        "validation_sha256": file_sha256(required["validation"]),
        "graph_sha256": graph_hashes,
    }

def main():
    args = utility.parser.parse_args()

    utility.tools.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    protocol_identity = build_protocol_identity(args)
    sealed_ledger_path = None
    if args.run_stage == "sealed_test":
        ledger_dir = Path("results") / "sealed_test_ledger" / args.dataset
        ledger_key = (
            f"{args.run_name}_seed{args.seed}_"
            f"{protocol_identity['split_manifest_sha256'][:16]}"
        )
        sealed_ledger_path = ledger_dir / f"{ledger_key}.json"
        if sealed_ledger_path.exists():
            raise FileExistsError(
                "This model/dataset/seed/split already has a sealed-test claim: "
                f"{sealed_ledger_path.resolve()}"
            )


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{timestamp}"


    checkpoint_dir = (
        Path("checkpoints")
        / args.dataset
        / run_id
    )

    result_dir = (
        Path("results")
        / args.dataset
        / run_id
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)


    best_checkpoint_path = (
        checkpoint_dir
        / f"{args.run_name}_best_validation.pt"
    )

    last_checkpoint_path = (
        checkpoint_dir
        / f"{args.run_name}_last.pt"
    )

    config_path = (
        result_dir
        / f"{args.run_name}_config.json"
    )

    metrics_path = (
        result_dir
        / f"{args.run_name}_metrics.csv"
    )

    route_path = (
        result_dir
        / f"{args.run_name}_amr_routes.csv"
    )

    console_path = (
        result_dir
        / f"{args.run_name}_console.log"
    )

    sealed_test_path = (
        result_dir
        / f"{args.run_name}_sealed_test_metrics.json"
    )


    console_file = open(
        console_path,
        "w",
        encoding="utf-8",
        buffering=1
    )


    original_stdout = sys.stdout
    original_stderr = sys.stderr


    sys.stdout = Tee(original_stdout, console_file)
    sys.stderr = Tee(original_stderr, console_file)

    print("=" * 80)


    config_record = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "requested_device": args.device,
        "resolved_device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "arguments": vars(args).copy(),
        "reporting_seeds": list(utility.parser.REPORTING_SEEDS),
        "frozen_split_seed": utility.parser.FROZEN_SPLIT_SEED,
        "run_stage": args.run_stage,
        "protocol_identity": protocol_identity,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as config_file:
        json.dump(
            config_record,
            config_file,
            ensure_ascii=False,
            indent=2
        )

    print("Configuration saved:", config_path)

    top_k = ast.literal_eval(args.top_K)
    if 20 not in top_k:
        raise ValueError("Revision protocol requires 20 in --top_K.")
    selection_index = top_k.index(20)


    metric_columns = [
        "epoch",
        "train_total_loss",
        "bpr_loss",
        "reg_loss",
        "hdcm_fv_loss",
        "hdcm_ii_loss",
        "vib_loss",
        "attr_recon_loss",
        "edge_recon_loss",
        "page_projection_count",
        "page_shared_gradient_count",
        "page_conflict_rate",
    ]

    for k in top_k:
        metric_columns.extend([
            f"validation_recall@{k}",
            f"validation_ndcg@{k}",
            f"validation_precision@{k}",
        ])


    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=metric_columns
        )
        writer.writeheader()

    print("Metrics file created:", metrics_path)

    print("Run ID:", run_id)
    print("Dataset:", args.dataset)
    print("Device:", device)
    print("Checkpoint directory:", checkpoint_dir)
    print("Result directory:", result_dir)
    print("=" * 80)


    # load train and test
    dataset = Data(str(Path(args.data_path) / args.dataset), args)
    # load hete model
    uu_graph, ii_graph = load_data(
        args.dataset,
        device,
        data_root=args.data_path,
    )

    if args.run_stage == "train_validate":
        if dataset.num_test != 0 or dataset.test_dict:
            raise RuntimeError(
                "train_validate must not load test interactions."
            )

    config_record["data_access"] = {
        "num_train_loaded": int(dataset.num_train),
        "num_validation_loaded": int(dataset.num_validation),
        "num_test_loaded": int(dataset.num_test),
        "test_loaded": bool(dataset.test_dict),
    }
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_record, config_file, ensure_ascii=False, indent=2)

    model = AMDRRec(args, dataset, uu_graph, ii_graph, device=device)

    model = model.to(device)

    user_path_names, item_path_names = get_meta_path_names(args.dataset)
    route_columns = [
        "epoch",
        "side",
        "path_names",
        "path_weights",
        "top2_indices",
        "top2_names",
        "selection_changed",
        "cumulative_switches",
        "mean_path_logit_gradients",
        "unselected_nonzero_gradient",
    ]
    previous_top2 = {"user": None, "item": None}
    cumulative_switches = {"user": 0, "item": 0}

    if args.amr_mode == "global" and args.run_stage != "sealed_test":
        with open(route_path, "w", newline="", encoding="utf-8") as route_file:
            csv.DictWriter(route_file, fieldnames=route_columns).writeheader()

        with torch.no_grad():
            initial_routes = {
                "user": model.user_amr.get_hard_topk_route(
                    model.user_embedding.weight,
                    len(model.uu_graph),
                ),
                "item": model.item_amr.get_hard_topk_route(
                    model.item_embedding.weight,
                    len(model.ii_graph),
                ),
            }
        with open(route_path, "a", newline="", encoding="utf-8") as route_file:
            writer = csv.DictWriter(route_file, fieldnames=route_columns)
            for side, names in (
                ("user", user_path_names),
                ("item", item_path_names),
            ):
                row, previous_top2[side], cumulative_switches[side] = (
                    build_amr_route_row(
                        epoch=0,
                        side=side,
                        path_names=names,
                        route=initial_routes[side],
                        mean_path_logit_gradients=None,
                        previous_top2=previous_top2[side],
                        cumulative_switches=cumulative_switches[side],
                    )
                )
                writer.writerow(row)
        print("AMR route audit file created:", route_path)

    if args.run_stage == "sealed_test":
        frozen_checkpoint = Path(args.frozen_checkpoint).resolve()
        if not frozen_checkpoint.is_file():
            raise FileNotFoundError(
                f"Frozen validation checkpoint does not exist: {frozen_checkpoint}"
            )
        checkpoint = torch.load(str(frozen_checkpoint), map_location=device)
        if checkpoint.get("dataset") != args.dataset:
            raise ValueError(
                "Frozen checkpoint dataset does not match the requested dataset."
            )
        if checkpoint.get("selection_metric_name") != "validation_ndcg@20":
            raise ValueError(
                "Checkpoint was not selected by validation_ndcg@20."
            )
        checkpoint_arguments = checkpoint.get("arguments", {})
        if checkpoint_arguments.get("seed") != args.seed:
            raise ValueError("Frozen checkpoint seed does not match --seed.")
        if checkpoint_arguments.get("run_name") != args.run_name:
            raise ValueError("Frozen checkpoint run_name does not match --run_name.")
        if checkpoint.get("protocol_identity") != protocol_identity:
            raise ValueError(
                "Frozen checkpoint split/graph identity does not match the active protocol."
            )
        restore_result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        if restore_result.missing_keys or restore_result.unexpected_keys:
            raise RuntimeError(f"Strict checkpoint restore failed: {restore_result}")

        ledger_dir = sealed_ledger_path.parent
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = sealed_ledger_path
        claim = {
            "status": "claimed_before_evaluation",
            "dataset": args.dataset,
            "seed": args.seed,
            "run_name": args.run_name,
            "split_manifest_sha256": protocol_identity["split_manifest_sha256"],
            "frozen_checkpoint": str(frozen_checkpoint),
            "frozen_checkpoint_sha256": file_sha256(frozen_checkpoint),
            "claimed_at": datetime.now().isoformat(),
        }
        with open(ledger_path, "x", encoding="utf-8") as ledger_file:
            json.dump(claim, ledger_file, ensure_ascii=False, indent=2)

        test_result = utility.batch_test.Test(dataset, model, device, args)
        sealed_record = {
            "run_id": run_id,
            "dataset": args.dataset,
            "seed": args.seed,
            "run_stage": "sealed_test",
            "test_evaluations": 1,
            "frozen_checkpoint": str(frozen_checkpoint),
            "frozen_checkpoint_sha256": file_sha256(frozen_checkpoint),
            "validation_selected_epoch": checkpoint.get("epoch"),
            "validation_selection_metric": checkpoint.get("selection_metric_name"),
            "validation_selection_value": checkpoint.get("selection_metric_value"),
            "test_metrics": {
                key: value.tolist()
                for key, value in test_result.items()
            },
            "protocol_identity": protocol_identity,
            "sealed_test_ledger": str(ledger_path.resolve()),
        }
        with open(sealed_test_path, "x", encoding="utf-8") as output_file:
            json.dump(sealed_record, output_file, ensure_ascii=False, indent=2)
        completed_claim = dict(claim)
        completed_claim.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "sealed_test_record": str(sealed_test_path.resolve()),
                "test_evaluations": 1,
            }
        )
        ledger_path.write_text(
            json.dumps(completed_claim, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Sealed test completed exactly once.")
        print("Frozen checkpoint SHA-256:", sealed_record["frozen_checkpoint_sha256"])
        print("Sealed test metrics:", sealed_record["test_metrics"])
        print("Sealed test record:", sealed_test_path)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_file.close()
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_report_recall = 0.
    best_validation_ndcg = float("-inf")
    best_report_epoch = 0
    validation_checks_without_improvement = 0
    last_result = None
    last_completed_epoch = 0

    for epoch in range(args.epochs):
        last_completed_epoch = epoch + 1
        start_time = time()

        model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        batch_order_generator = torch.Generator(device="cpu")
        batch_order_generator.manual_seed(args.seed + epoch)
        batch_order = torch.randperm(
            users.size(0),
            generator=batch_order_generator,
        )
        users = users[batch_order]
        pos_items = pos_items[batch_order]
        neg_items = neg_items[batch_order]

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        num_batch = (len(users) + args.batch_size - 1) // args.batch_size
        total_loss_list = []
        page_projection_count = 0
        page_shared_gradient_count = 0
        route_gradient_sums = None
        route_gradient_counts = None
        if args.amr_mode == "global":
            route_gradient_sums = {
                "user": torch.zeros_like(model.user_amr.path_logits),
                "item": torch.zeros_like(model.item_amr.path_logits),
            }
            route_gradient_counts = {"user": 0, "item": 0}

        for batch_i, (batch_users, batch_positive, batch_negative) in \
                enumerate(utility.tools.mini_batch(users, pos_items, neg_items,
                                                   batch_size=int(args.batch_size))):
            loss_list = model(batch_users,batch_positive,batch_negative,epoch=epoch,)

            if batch_i == 0:
                assert len(loss_list) >= 1
                total_loss_list = [0.] * len(loss_list)

            total_loss = 0.
            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()

            primary_loss = (
                loss_list[0]
                + loss_list[1]
                + loss_list[2]
                + loss_list[3]
                + loss_list[4]
            )
            auxiliary_loss = (
                loss_list[5]
                + loss_list[6]
            )

            opt.zero_grad()
            page_statistics = apply_primary_anchored_backward(
                model=model,
                primary_loss=primary_loss,
                auxiliary_loss=auxiliary_loss,
            )
            page_projection_count += page_statistics["projection_count"]
            page_shared_gradient_count += page_statistics["both_used_count"]

            if args.amr_mode == "global":
                for side, router in (
                    ("user", model.user_amr),
                    ("item", model.item_amr),
                ):
                    if router.path_logits.grad is not None:
                        route_gradient_sums[side] += router.path_logits.grad.detach()
                        route_gradient_counts[side] += 1
            opt.step()

            # print("\t Batch:", batch_i, " loss_list:", loss_list, "total_loss:", total_loss, total_loss_list)

        end_time = time()

        loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) \
                    + " = " + " + ".join([str(round(i / num_batch, 6)) for i in total_loss_list])
        print("\t Epoch: %4d| train time: %.3f | train_loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if args.amr_mode == "global":
            with torch.no_grad():
                user_route = (
                    model.user_amr.get_hard_topk_route(
                        node_embeddings=model.user_embedding.weight,
                        num_paths=len(model.uu_graph),
                    )
                )
                item_route = (
                    model.item_amr.get_hard_topk_route(
                        node_embeddings=model.item_embedding.weight,
                        num_paths=len(model.ii_graph),
                    )
                )

            print(
                "\t User Router weights:",
                user_route[
                    "global_path_weights"
                ].detach().cpu().tolist(),
            )

            mean_route_gradients = {}
            for side in ("user", "item"):
                if route_gradient_counts[side] > 0:
                    mean_route_gradients[side] = (
                        route_gradient_sums[side] / route_gradient_counts[side]
                    )
                else:
                    mean_route_gradients[side] = None

            with open(route_path, "a", newline="", encoding="utf-8") as route_file:
                writer = csv.DictWriter(route_file, fieldnames=route_columns)
                for side, names, route in (
                    ("user", user_path_names, user_route),
                    ("item", item_path_names, item_route),
                ):
                    row, previous_top2[side], cumulative_switches[side] = (
                        build_amr_route_row(
                            epoch=epoch + 1,
                            side=side,
                            path_names=names,
                            route=route,
                            mean_path_logit_gradients=mean_route_gradients[side],
                            previous_top2=previous_top2[side],
                            cumulative_switches=cumulative_switches[side],
                        )
                    )
                    writer.writerow(row)
            print(
                "\t User Router Top-2:",
                user_route[
                    "topk_indices"
                ].detach().cpu().tolist(),
            )
            print(
                "\t Item Router weights:",
                item_route[
                    "global_path_weights"
                ].detach().cpu().tolist(),
            )
            print(
                "\t Item Router Top-2:",
                item_route[
                    "topk_indices"
                ].detach().cpu().tolist(),
            )

        should_evaluate = (
            args.run_stage == "train_validate"
            and (epoch + 1) % args.validation_interval == 0
        )
        if should_evaluate:
            if not args.sparsity_test:
                result = utility.batch_test.Validation(dataset, model, device, args)


                last_result = result


                average_losses = [
                    value / num_batch
                    for value in total_loss_list
                ]

                metric_row = {
                    "epoch": epoch + 1,
                    "train_total_loss": sum(average_losses),
                    "bpr_loss": average_losses[0],
                    "reg_loss": average_losses[1],
                    "hdcm_fv_loss": average_losses[2],
                    "hdcm_ii_loss": average_losses[3],
                    "vib_loss": average_losses[4],
                    "attr_recon_loss": (
                        average_losses[5]
                        if len(average_losses) > 5
                        else 0.0
                    ), 
                    "edge_recon_loss": (
                        average_losses[6]
                        if len(average_losses) > 6
                        else 0.0
                    ),
                    "page_projection_count": page_projection_count,
                    "page_shared_gradient_count": page_shared_gradient_count,
                    "page_conflict_rate": (
                        page_projection_count / page_shared_gradient_count
                        if page_shared_gradient_count > 0
                        else 0.0
                    ),
                }


                for index, k in enumerate(top_k):
                    metric_row[f"validation_recall@{k}"] = float(
                        result["recall"][index]
                    )
                    metric_row[f"validation_ndcg@{k}"] = float(
                        result["ndcg"][index]
                    )
                    metric_row[f"validation_precision@{k}"] = float(
                        result["precision"][index]
                    )


                with open(
                    metrics_path,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as metrics_file:
                    writer = csv.DictWriter(
                        metrics_file,
                        fieldnames=metric_columns
                    )
                    writer.writerow(metric_row)
                current_ndcg = float(result["ndcg"][selection_index])
                current_recall = float(result["recall"][selection_index])
                ndcg_improved = (
                    current_ndcg
                    > best_validation_ndcg + args.early_stop_min_delta
                )
                if ndcg_improved:
                    best_report_epoch = epoch + 1
                    best_validation_ndcg = current_ndcg
                    best_report_recall = current_recall
                    validation_checks_without_improvement = 0

                    save_checkpoint(
                        path=best_checkpoint_path,
                        epoch=epoch + 1,
                        model=model,
                        optimizer=opt,
                        args=args,
                        result=result,
                        selection_metric_name="validation_ndcg@20",
                        selection_metric_value=best_validation_ndcg,
                        protocol_identity=protocol_identity,
                    )

                    print(
                        "\t Best-validation checkpoint saved:",
                        best_checkpoint_path
                    )
                else:
                    validation_checks_without_improvement += 1

                print(
                    "\t Validation Recall:",
                    result['recall'],
                    "Validation NDCG:",
                    result['ndcg'],
                )
                if (
                    validation_checks_without_improvement
                    >= args.early_stop_patience
                ):
                    print(
                        "\t Early stopping triggered after",
                        validation_checks_without_improvement,
                        "validation checks without improvement.",
                    )
                    break
            else:
                raise RuntimeError(
                    "Sparsity testing is forbidden during train_validate. "
                    "Run it only as a separate sealed analysis after model selection."
                )

        save_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch + 1,
            model=model,
            optimizer=opt,
            args=args,
            result=last_result,
            selection_metric_name=(
                "validation_ndcg@20"
                if args.run_stage == "train_validate"
                else "none"
            ),
            selection_metric_value=(
                best_validation_ndcg
                if args.run_stage == "train_validate"
                else float("nan")
            ),
            protocol_identity=protocol_identity,
        )
    # vis_embedding
    # user_emb = torch.tensor(model.user_embedding.weight, dtype=torch.float32)
    # item_emb = torch.tensor(model.item_embedding.weight, dtype=torch.float32)
    # all_embedding = torch.cat([user_emb, item_emb])
    # torch.save(all_embedding, 'data/emb/all_embedding.pth')
    # torch.save(user_emb, 'data/dbook_user_emb.pth')
    # torch.save(item_emb, 'data/dbook_item_emb.pth')

    # vis_noise
    # hete_user_embedding = []
    # hete_item_embedding = []
    # for i in range(len(uu_graph)):
    #     hete_user_embedding.append(model.user_autoencoder[i](model.uu_graph[i], user_emb).flatten(1))
    # for i in range(len(ii_graph)):
    #     hete_item_embedding.append(model.item_autoencoder[i](model.ii_graph[i], item_emb).flatten(1))
    # torch.save(hete_user_embedding[0], 'data/emb/dbmovie_user_noise_uu.pth')
    # torch.save(hete_user_embedding[1], 'data/emb/dbmovie_user_noise_ugu.pth')
    # torch.save(hete_item_embedding[0], 'data/emb/dbmovie_item_noise_mam.pth')
    # torch.save(hete_item_embedding[1], 'data/emb/dbmovie_item_noise_mdm.pth')

    print("\t Model training process completed.")
    print("\t best epoch:", best_report_epoch)
    print("\t best validation NDCG@20:", best_validation_ndcg)
    print("\t paired validation Recall@20:", best_report_recall)
    print("\t last completed epoch:", last_completed_epoch)
    if last_result is None:
        print("\t last validation recall: not evaluated")
    else:
        print(
            "\t last validation Recall@20:",
            float(last_result["recall"][selection_index]),
        )
    
    print("\t Results saved successfully.")
    print("\t Config:", config_path)
    print("\t Metrics:", metrics_path)
    if args.amr_mode == "global" and args.run_stage != "sealed_test":
        print("\t AMR route audit:", route_path)
    print("\t Console log:", console_path)
    print("\t Best-validation checkpoint:", best_checkpoint_path)
    print("\t Last checkpoint:", last_checkpoint_path)


    console_file.flush()


    sys.stdout = original_stdout
    sys.stderr = original_stderr


    console_file.close()
if __name__ == '__main__':
    main()
