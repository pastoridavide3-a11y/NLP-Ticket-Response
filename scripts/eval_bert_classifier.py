"""Load saved BERT classifiers and compute metrics + confusion matrix on the test set."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.classification.transformer_clf import predict_bert
from src.utils import io as uio
from src.utils.plotting import save_fig


def _confusion_fig(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), max(5, 0.5 * len(labels))))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    save_fig(fig, path)


def main():
    test = uio.load_df(config.DATA_PROCESSED / "test.csv")

    targets = {
        "priority":   (config.TABLES / "classification" / "bert_priority",   "priority", config.PRIORITIES),
        "department": (config.TABLES / "classification" / "bert_department",  "queue",    config.DEPARTMENTS),
    }

    for target_name, (model_dir, col, labels) in targets.items():
        print(f"\n=== BERT {target_name} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        id2label = model.config.id2label

        preds = predict_bert(model, tokenizer, id2label, test["ticket"].tolist())
        true  = test[col].tolist()

        report = classification_report(true, preds, labels=labels, zero_division=0)
        summary = {
            "accuracy":    round(accuracy_score(true, preds), 4),
            "f1_macro":    round(f1_score(true, preds, average="macro",    zero_division=0), 4),
            "f1_weighted": round(f1_score(true, preds, average="weighted", zero_division=0), 4),
        }

        print(report)
        (model_dir / "test_report.txt").write_text(report, encoding="utf-8")
        uio.save_json(summary, model_dir / "test_summary.json")

        fig_dir = config.FIGURES / "classification" / f"bert_{target_name}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _confusion_fig(true, preds, labels, f"BERT {target_name} — test", fig_dir / "test_confusion.png")

        print(f"saved to {model_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
