#!/usr/bin/env python3
"""
main.py
-------
Entry point for the ML Algorithm Selector project.

Default behavior (no arguments):
    Loads the built-in Iris dataset, evaluates 7 candidate ML algorithms
    using Stratified 5-Fold Cross Validation, prints a ranked results
    table, saves it to output/results.csv, and prints a LangChain-powered
    natural-language recommendation of the best model.

Using your OWN dataset instead of Iris:
    python main.py --csv path/to/your_data.csv --target your_label_column

Examples:
    python main.py
    python main.py --folds 10
    python main.py --csv data/my_dataset.csv --target species
    python main.py --csv data/my_dataset.csv --target species --dataset-name "My Dataset"
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_dataset          # noqa: E402
from models import get_candidate_models        # noqa: E402
from evaluator import evaluate_models, get_best_model_name  # noqa: E402
from langchain_report import generate_recommendation         # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ML algorithms with 5-fold CV (Iris by default, "
        "or bring your own CSV dataset)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a custom CSV dataset. If omitted, the built-in Iris "
        "dataset is used.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target/label column in your CSV (required if "
        "--csv is provided).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Friendly name used in the report (default: 'Iris' or the CSV "
        "filename).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5, as requested).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where results.csv and report.txt will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load data (Iris by default, or a user-supplied CSV)
    # ------------------------------------------------------------------
    X, y, feature_names, target_names = load_dataset(
        csv_path=args.csv, target_column=args.target
    )
    dataset_name = args.dataset_name or (
        os.path.splitext(os.path.basename(args.csv))[0] if args.csv else "Iris"
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Samples: {len(y)}, Features: {len(feature_names)}, "
          f"Classes: {len(target_names)} {target_names}")

    # ------------------------------------------------------------------
    # 2. Define candidate ML algorithms
    # ------------------------------------------------------------------
    models = get_candidate_models()
    print(f"\nCandidate algorithms ({len(models)}): {', '.join(models.keys())}")

    # ------------------------------------------------------------------
    # 3. Evaluate each algorithm with Stratified K-Fold Cross Validation
    # ------------------------------------------------------------------
    print(f"\nRunning Stratified {args.folds}-Fold Cross Validation ...\n")
    results_df = evaluate_models(models, X, y, n_splits=args.folds)

    display_cols = [
        "rank", "model", "accuracy_mean", "accuracy_std",
        "f1_macro_mean", "precision_macro_mean", "recall_macro_mean",
        "fit_time_mean_sec",
    ]
    print(results_df[display_cols].to_string(index=False, float_format="%.4f"))

    best_model = get_best_model_name(results_df)
    print(f"\n>>> Best performing model (by mean accuracy): {best_model}")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    # ------------------------------------------------------------------
    # 5. Generate a natural-language recommendation via LangChain
    # ------------------------------------------------------------------
    print("\nGenerating recommendation report...\n")
    report = generate_recommendation(results_df, dataset_name=dataset_name)

    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print("=" * 70)
    print("RECOMMENDATION REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
