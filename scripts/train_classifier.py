"""Train TF-IDF + LogReg classifiers for priority and department."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config
from src.classification.baseline_logreg import build_clf
from src.classification.evaluate import eval_classifier
from src.utils import io as uio
from src.utils.plotting import save_fig
from src.utils.seeds import set_seed

TARGETS = {
    "priority":   ("priority", config.PRIORITIES),
    "department": ("queue",    config.DEPARTMENTS),
}


def _confusion_fig(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), max(5, 0.5 * len(labels))))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    save_fig(fig, path)


def main() -> None:
    set_seed(config.SEED)
    train = uio.load_df(config.DATA_PROCESSED / "train.csv")
    test  = uio.load_df(config.DATA_PROCESSED / "test.csv")

    clf_out_dir = config.TABLES / "classification"
    clf_fig_dir = config.FIGURES / "classification"

    summaries = []
    for name, (col, labels) in TARGETS.items():
        print(f"[clf] training target={name}")
        model = build_clf(seed=config.SEED)
        model.fit(train["ticket"].tolist(), train[col].tolist())

        preds = model.predict(test["ticket"].tolist())
        summary = eval_classifier(test[col].tolist(), preds, labels, out_dir=clf_out_dir / name, name="test")
        summary["target"] = name
        summaries.append(summary)

        _confusion_fig(test[col].tolist(), preds, labels, f"{name} — test", clf_fig_dir / f"{name}_test_confusion.png")

        joblib.dump(model, config.MODELS / f"classifier_{name}.joblib")

    uio.save_df(pd.DataFrame(summaries), clf_out_dir / "summary.csv")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
