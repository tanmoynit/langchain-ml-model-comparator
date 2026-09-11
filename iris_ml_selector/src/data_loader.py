"""
data_loader.py
---------------
Responsible for loading a dataset for the model-selection pipeline.

By default it loads the classic Iris dataset that ships with scikit-learn,
so the project runs "out of the box" with zero external files.

To use your OWN dataset in the future, you have two options:

1) Point to a CSV file:
   python main.py --csv path/to/your_data.csv --target target_column_name

2) Edit config.py and change DEFAULT_CSV_PATH / DEFAULT_TARGET_COLUMN.

The loader returns a clean (X, y, feature_names, target_names) tuple no
matter which source is used, so nothing downstream (models.py,
evaluator.py, langchain_report.py) needs to know or care where the data
came from.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


def load_builtin_iris():
    """Load the classic Iris dataset bundled with scikit-learn."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = list(X.columns)
    target_names = list(iris.target_names)
    return X, y, feature_names, target_names


def load_csv_dataset(csv_path: str, target_column: str):
    """
    Load any tabular dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file. Each row = one sample, each column = one feature
        (plus the target/label column somewhere in the file).
    target_column : str
        Name of the column in the CSV that holds the class label
        (the thing you want to predict).

    Returns
    -------
    X : pd.DataFrame of features (numeric columns only; non-numeric feature
        columns are dropped with a warning, since the demo models expect
        numeric input. Extend this function if you need categorical
        encoding.)
    y : pd.Series of encoded integer labels
    feature_names : list[str]
    target_names : list[str]  (original class label names, order matches the
        encoded integer classes)
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {csv_path}. "
            f"Available columns are: {list(df.columns)}"
        )

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    # Keep only numeric feature columns for this generic pipeline.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(
            f"[data_loader] Warning: dropping non-numeric feature columns "
            f"{non_numeric}. Encode/engineer them first if you need them."
        )
        X = X.drop(columns=non_numeric)

    if X.empty:
        raise ValueError(
            "No numeric feature columns remain after dropping non-numeric "
            "columns. Please encode categorical features before running "
            "this pipeline."
        )

    # Encode the target labels to integers 0..n_classes-1 (works whether
    # the original labels were strings or numbers already).
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw), name=target_column)
    target_names = [str(c) for c in encoder.classes_]

    feature_names = list(X.columns)
    return X, y, feature_names, target_names


def load_dataset(csv_path: str | None = None, target_column: str | None = None):
    """
    Single entry point used by main.py.

    If csv_path is None -> use the built-in Iris dataset.
    Otherwise -> load the user's CSV using target_column as the label.
    """
    if csv_path is None:
        print("[data_loader] Using built-in Iris dataset.")
        return load_builtin_iris()

    if target_column is None:
        raise ValueError(
            "You provided --csv but not --target. Please specify which "
            "column is the label, e.g. --target species"
        )

    print(f"[data_loader] Loading custom dataset from: {csv_path}")
    return load_csv_dataset(csv_path, target_column)
