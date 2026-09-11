"""
evaluator.py
------------
Runs Stratified 5-Fold Cross Validation over every candidate model and
returns a tidy results table (pandas DataFrame) that can be printed,
saved to CSV, and/or handed to the LangChain report generator.

Stratified K-Fold is used (rather than plain KFold) because it preserves
the class distribution in every fold, which matters for classification
tasks like Iris (3 balanced classes) and even more for imbalanced custom
datasets a user might bring in later.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_models(
    models: dict,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: tuple = ("accuracy", "f1_macro", "precision_macro", "recall_macro"),
) -> pd.DataFrame:
    """
    Parameters
    ----------
    models : dict {name: estimator}
        Output of models.get_candidate_models()
    X, y : features / labels
    n_splits : int
        Number of folds for cross validation (5 = "5-fold" as requested).
    scoring : tuple of sklearn scoring metric names.

    Returns
    -------
    pd.DataFrame sorted by mean accuracy (descending), with columns:
        model, accuracy_mean, accuracy_std, f1_macro_mean, ...,
        fit_time_mean
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, estimator in models.items():
        start = time.time()
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        elapsed = time.time() - start

        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = np.mean(scores[key])
            row[f"{metric}_std"] = np.std(scores[key])
        row["fit_time_mean_sec"] = np.mean(scores["fit_time"])
        row["total_eval_time_sec"] = elapsed
        rows.append(row)

    results_df = pd.DataFrame(rows)
    primary_metric = f"{scoring[0]}_mean"
    results_df = results_df.sort_values(by=primary_metric, ascending=False).reset_index(
        drop=True
    )
    results_df.insert(0, "rank", results_df.index + 1)
    return results_df


def get_best_model_name(results_df: pd.DataFrame, metric: str = "accuracy_mean") -> str:
    """Convenience helper: returns the model name with the highest score
    on the given metric column."""
    best_row = results_df.sort_values(by=metric, ascending=False).iloc[0]
    return best_row["model"]
