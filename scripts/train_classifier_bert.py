"""Fine-tune DistilBERT for priority and department classification.

This script was run on the Bocconi HPC cluster (GPU). To reproduce:
    python scripts/train_classifier_bert.py

Results are saved under /home/3380930/output/ on the cluster, then copied
locally to results/models/bert_priority/ and results/models/bert_department/.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from src import config


SMALL_DEPTS = {
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
}


def load_data():
    print("Carico il dataset...")
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = ds.to_pandas()
    df = df[df["language"] == "en"].rename(columns={"body": "ticket"})
    df = df[["ticket", "answer", "priority", "queue"]].dropna().reset_index(drop=True)

    df = df[df["priority"].isin(config.PRIORITIES)]
    df["queue"] = df["queue"].apply(lambda q: "Other" if q in SMALL_DEPTS else q)
    df = df[df["queue"].isin(config.DEPARTMENTS)]

    strat = df["priority"] + "_" + df["queue"]
    train, temp = train_test_split(df, test_size=0.30, stratify=strat, random_state=config.SEED)
    strat2 = temp["priority"] + "_" + temp["queue"]
    val, test = train_test_split(temp, test_size=0.50, stratify=strat2, random_state=config.SEED)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def predict(model, tokenizer, id2label, texts, batch_size=64):
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding="max_length",
                            truncation=True, max_length=128, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return [id2label[p] for p in preds]


def train_bert(train_texts, train_labels, val_texts, val_labels,
               epochs=15, batch_size=32, lr=2e-5, patience=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    label_list = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    labels_array = np.array([label2id[l] for l in train_labels])

    class_weights = compute_class_weight("balanced",
                                         classes=np.arange(len(label_list)),
                                         y=labels_array)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tokenizer(train_texts, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"],
                            torch.tensor(labels_array))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(label_list),
        id2label=id2label, label2id=label2id).to(device)

    total_steps = len(loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    best_f1, best_state, no_improve = 0, None, 0
    for epoch in range(epochs):
        model.train()
        total = 0
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()

        preds_val = predict(model, tokenizer, id2label, val_texts)
        val_f1 = f1_score(val_labels, preds_val, average="macro", zero_division=0)
        print(f"epoch {epoch+1}/{epochs}  loss={total/len(loader):.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping a epoch {epoch+1}, best val_f1={best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return model, tokenizer, id2label


def save_results(preds, true_labels, label_names, task_name, out_base):
    out_base = Path(out_base)
    (out_base / "metrics").mkdir(parents=True, exist_ok=True)
    (out_base / "figures").mkdir(parents=True, exist_ok=True)

    report = classification_report(true_labels, preds, zero_division=0)
    print(report)
    (out_base / "metrics" / "test_report.txt").write_text(report, encoding="utf-8")

    with open(out_base / "metrics" / "test_summary.json", "w") as f:
        json.dump({
            "accuracy":    round(accuracy_score(true_labels, preds), 4),
            "f1_macro":    round(f1_score(true_labels, preds, average="macro",    zero_division=0), 4),
            "f1_weighted": round(f1_score(true_labels, preds, average="weighted", zero_division=0), 4),
        }, f, indent=2)

    cm = confusion_matrix(true_labels, preds, labels=label_names, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"BERT {task_name} — test")
    plt.tight_layout()
    fig.savefig(out_base / "figures" / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"Salvato in {out_base}")


def main():
    train, val, test = load_data()

    targets = {
        "bert_priority":   ("priority", config.PRIORITIES),
        "bert_department": ("queue",    config.DEPARTMENTS),
    }

    for task_name, (col, labels) in targets.items():
        print("=" * 50 + f" {task_name.upper()}")
        model, tokenizer, id2label = train_bert(
            train["ticket"].tolist(), train[col].tolist(),
            val["ticket"].tolist(),   val[col].tolist(),
        )
        preds = predict(model, tokenizer, id2label, test["ticket"].tolist())
        out_base = config.MODELS / task_name
        save_results(preds, test[col].tolist(), labels, task_name, out_base)
        model.save_pretrained(out_base / "model")
        tokenizer.save_pretrained(out_base / "model")

    print("DONE")


if __name__ == "__main__":
    main()
