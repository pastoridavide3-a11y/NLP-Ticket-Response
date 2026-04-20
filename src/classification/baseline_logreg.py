"""TF-IDF + Logistic Regression classifier.

Strong linear baseline. Class weighting compensates for imbalance.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.tfidf import build_tfidf


def build_clf(
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    seed: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_tfidf()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
