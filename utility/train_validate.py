from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utility.parser import PAPER_DATA_ROOT, REPORTING_SEEDS


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen five reporting seeds sequentially in train_validate stage. "
            "Arguments after -- are forwarded to main.py."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", default=PAPER_DATA_ROOT)
    parser.add_argument("--run_name", default="AMDRRec")
    args, forwarded = parser.parse_known_args()
    forbidden = {"--seed", "--run_stage", "--frozen_checkpoint", "--run_mode"}
    conflicts = [value for value in forwarded if value in forbidden]
    if conflicts:
        parser.error(f"Do not override protocol-controlled arguments: {conflicts}")
    return args, forwarded


def main() -> int:
    args, forwarded = parse_args()
    code_root = Path(__file__).resolve().parent.parent
    main_path = code_root / "main.py"
    for seed in REPORTING_SEEDS:
        seed_run_name = f"{args.run_name}_seed{seed}"
        command = [
            sys.executable,
            str(main_path),
            "--dataset",
            args.dataset,
            "--data_path",
            args.data_path,
            "--run_stage",
            "train_validate",
            "--seed",
            str(seed),
            "--run_name",
            seed_run_name,
            *forwarded,
        ]
        print("=" * 80, flush=True)
        print(f"Starting frozen reporting seed: {seed}", flush=True)
        completed = subprocess.run(command, cwd=code_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Seed {seed} failed with exit code {completed.returncode}; "
                "remaining seeds were not started."
            )
    print("All five train_validate runs completed.")
    print("No sealed test was executed by this launcher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
