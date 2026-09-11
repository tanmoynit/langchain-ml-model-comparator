"""
models.py
---------
Defines the pool of candidate ML algorithms that will be compared using
5-fold cross validation. Add / remove / tune models here — everything
else in the pipeline (evaluator.py, main.py) will automatically pick up
whatever is returned by get_candidate_models().
"""

from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_candidate_models(random_state: int = 42) -> dict:
    """
    Returns a dict of {model_name: sklearn_estimator_or_pipeline}.

    Models that are sensitive to feature scale (Logistic Regression, SVM,
    KNN) are wrapped in a Pipeline with StandardScaler so scaling happens
    correctly *inside* each cross-validation fold (avoids data leakage).
    Tree-based models don't need scaling, so they're left as-is.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }
    return models
