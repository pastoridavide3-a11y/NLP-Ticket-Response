"""Phase 3: train priority + department classifiers. TF-IDF + LogReg.

Saves metrics, confusion matrices, and per-split predictions per target.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.classification.baseline_logreg import build_clf
from src.classification.evaluate import eval_classifier
from src.utils import io as uio
from src.utils.plotting import save_fig
from src.utils.seeds import set_seed


TARGETS = {
    "priority": ("priority", config.PRIORITIES),
    "department": ("queue", config.DEPARTMENTS),
}


def _confusion_fig(y_true, y_pred, labels, title, path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), max(5, 0.5 * len(labels))))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    save_fig(fig, path)


def main() -> None:
    set_seed(config.SEED)
    train = uio.load_df(config.DATA_PROCESSED / "train.csv")
    val = uio.load_df(config.DATA_PROCESSED / "val.csv")
    test = uio.load_df(config.DATA_PROCESSED / "test.csv")

    clf_out_dir = config.TABLES / "classification"
    clf_fig_dir = config.FIGURES / "classification"
    clf_out_dir.mkdir(parents=True, exist_ok=True)
    clf_fig_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for name, (col, labels) in TARGETS.items():
        print(f"[clf] training target={name} col={col} n_labels={len(labels)}")
        model = build_clf(seed=config.SEED)
        model.fit(train["ticket"].tolist(), train[col].tolist())

        for split_name, df in (("val", val), ("test", test)):
            preds = model.predict(df["ticket"].tolist())
            summary = eval_classifier(
                df[col].tolist(), preds, labels,
                out_dir=clf_out_dir / name, name=split_name,
            )
            summary["target"] = name
            summary["split"] = split_name
            all_summaries.append(summary)
            _confusion_fig(
                df[col].tolist(), preds, labels,
                f"{name} — {split_name}",
                clf_fig_dir / f"{name}_{split_name}_confusion.png",
            )
            pred_df = df[["id", col]].copy()
            pred_df.columns = ["id", "true"]
            pred_df["pred"] = preds
            uio.save_df(pred_df, config.GENERATIONS / f"clf_{name}_{split_name}.parquet")

        joblib.dump(model, config.MODELS / f"classifier_{name}.joblib")

    uio.save_df(pd.DataFrame(all_summaries), clf_out_dir / "summary.csv")
    print(pd.DataFrame(all_summaries))


if __name__ == "__main__":
    main()
