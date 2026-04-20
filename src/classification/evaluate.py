"""Classification evaluation utilities."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.utils import io as uio


def eval_classifier(y_true, y_pred, labels, out_dir: Path, name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    precs, recs, f1s, sups = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"label": labels, "precision": precs, "recall": recs, "f1": f1s, "support": sups}
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uio.save_df(per_class, out_dir / f"{name}_per_class.csv")
    uio.save_df(cm_df.reset_index().rename(columns={"index": "label"}), out_dir / f"{name}_confusion.csv")
    (out_dir / f"{name}_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "name": name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n": int(len(y_true)),
    }
    uio.save_json(summary, out_dir / f"{name}_summary.json")
    return summary
