"""Fine-tune DistilBERT for priority and department classification.

Usage:
    python scripts/train_classifier_bert.py
    python scripts/train_classifier_bert.py --epochs 2 --batch 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config
from src.classification.evaluate import eval_classifier
from src.classification.transformer_clf import train_bert, predict_bert
from src.utils import io as uio
from src.utils.plotting import save_fig
from src.utils.seeds import set_seed


def main(epochs, batch, lr):
    set_seed(config.SEED)

    train = uio.load_df(config.DATA_PROCESSED / "train.csv")
    val   = uio.load_df(config.DATA_PROCESSED / "val.csv")
    test  = uio.load_df(config.DATA_PROCESSED / "test.csv")

    targets = {
        "priority":   ("priority", config.PRIORITIES),
        "department": ("queue",    config.DEPARTMENTS),
    }

    all_summaries = []

    for target_name, (col, labels) in targets.items():
        print(f"\n=== BERT — {target_name} ===")

        model, tokenizer, id2label = train_bert(
            train["ticket"].tolist(),
            train[col].tolist(),
            epochs=epochs, batch_size=batch, lr=lr, seed=config.SEED,
        )

        out_tables = config.TABLES  / "classification_bert" / target_name
        out_figs   = config.FIGURES / "classification_bert"
        out_tables.mkdir(parents=True, exist_ok=True)
        out_figs.mkdir(parents=True, exist_ok=True)

        for split_name, df in [("val", val), ("test", test)]:
            preds = predict_bert(model, tokenizer, id2label, df["ticket"].tolist(), batch_size=batch)

            summary = eval_classifier(df[col].tolist(), preds, labels, out_dir=out_tables, name=split_name)
            summary["target"] = target_name
            summary["split"]  = split_name
            all_summaries.append(summary)

            # confusion matrix plot
            cm = confusion_matrix(df[col].tolist(), preds, labels=labels, normalize="true")
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), max(5, 0.5 * len(labels))))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"BERT {target_name} — {split_name}")
            save_fig(fig, out_figs / f"{target_name}_{split_name}_confusion.png")

            pred_df = df[["id", col]].copy().rename(columns={col: "true"})
            pred_df["pred"] = preds
            uio.save_df(pred_df, config.GENERATIONS / f"bert_clf_{target_name}_{split_name}.parquet")

        model.save_pretrained(config.MODELS / f"bert_{target_name}")
        tokenizer.save_pretrained(config.MODELS / f"bert_{target_name}")

    summary_df = pd.DataFrame(all_summaries)
    uio.save_df(summary_df, config.TABLES / "classification_bert" / "summary.csv")
    print("\n--- results ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--lr",     type=float, default=2e-5)
    args = parser.parse_args()
    main(args.epochs, args.batch, args.lr)
