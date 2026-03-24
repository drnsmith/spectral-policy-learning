"""
scripts/run_experiment.py

Main entry point for spectral-policy-learning experiments.

Usage examples
--------------
# Full ablation grid (all classifiers × reward metrics × split strategies):
    python scripts/run_experiment.py --mode ablation

# Single condition:
    python scripts/run_experiment.py \
        --mode single \
        --classifier plsda \
        --reward_metric auc \
        --split_strategy acquisition

# Reproduce leaky baseline only:
    python scripts/run_experiment.py \
        --mode single \
        --split_strategy random \
        --classifier plsda \
        --reward_metric accuracy

Flags
-----
--data_path       path to raman_spectra_api_compounds.csv
--reward_metric   auc | accuracy | f1
--split_strategy  acquisition | random
--classifier      plsda | svm | cnn
--n_episodes      RL training episodes (default 200)
--max_steps       sub-policy steps per episode (default 5)
--seed            RNG seed (default 42)
--output_dir      where to save results (default results/)
--mode            single | ablation
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Make sure project root is on path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ablation import run_single, run_ablation_grid


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str):
    """
    Load and validate the Raman spectra CSV.

    Expected format:
      Rows    = spectra (3510)
      Columns = wavenumbers as floats (150.0 … 3425.0) + 'label' column
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the CSV.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    print(f"  Spectra:     {X.shape[0]}")
    print(f"  Wavenumbers: {X.shape[1]}")
    print(f"  Classes:     {len(np.unique(y))} ({list(np.unique(y))[:5]}...)")
    return X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Spectral Policy Learning Experiments"
    )
    p.add_argument(
        "--data_path",
        default="data/raw/raman_spectra_api_compounds.csv",
        help="Path to Raman spectra CSV",
    )
    p.add_argument(
        "--mode",
        choices=["single", "ablation"],
        default="ablation",
        help="Run a single condition or full ablation grid",
    )
    # Single-condition flags
    p.add_argument(
        "--classifier",
        choices=["plsda", "svm", "cnn"],
        default="plsda",
    )
    p.add_argument(
        "--reward_metric",
        choices=["auc", "accuracy", "f1"],
        default="auc",
    )
    p.add_argument(
        "--split_strategy",
        choices=["acquisition", "random"],
        default="acquisition",
    )
    # Training flags
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--max_steps",  type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", default="results/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    X, y = load_data(args.data_path)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        print(f"\nRunning single condition:")
        print(f"  Classifier:     {args.classifier}")
        print(f"  Reward metric:  {args.reward_metric}")
        print(f"  Split strategy: {args.split_strategy}")
        print(f"  Episodes:       {args.n_episodes}")
        print("=" * 60)

        result = run_single(
            X, y,
            classifier_name=args.classifier,
            reward_metric=args.reward_metric,
            split_strategy=args.split_strategy,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=True,
        )

        out_path = (
            f"{args.output_dir}/"
            f"{args.classifier}_{args.reward_metric}_{args.split_strategy}.json"
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult saved → {out_path}")

    elif args.mode == "ablation":
        results = run_ablation_grid(
            X, y,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"\n{'='*60}")
        print("ABLATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Classifier':<12} {'Reward':<12} {'Split':<14} {'AUC':<8} {'Acc':<8} {'F1':<8}")
        print("-" * 62)
        for r in results:
            cfg = r["config"]
            m   = r["test_metrics"]
            print(
                f"{cfg['classifier']:<12} "
                f"{cfg['reward_metric']:<12} "
                f"{cfg['split_strategy']:<14} "
                f"{m.get('auc', 0):.4f}   "
                f"{m.get('accuracy', 0):.4f}   "
                f"{m.get('f1', 0):.4f}"
            )


if __name__ == "__main__":
    main()
