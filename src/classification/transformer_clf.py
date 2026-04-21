"""DistilBERT fine-tuning helpers — plain functions, no classes.

Training settings used to produce the results in results/tables/classification/bert_*/:
- model: distilbert-base-uncased
- epochs: up to 15 with early stopping (patience=3) on val macro-F1
- batch_size: 32
- lr: 2e-5 with linear warmup (10%) and decay
- loss: weighted cross-entropy (balanced class weights)
- max_len: 128 tokens
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


def predict_bert(model, tokenizer, id2label: dict, texts: list[str],
                 max_len: int = 128, batch_size: int = 64) -> list[str]:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i+batch_size], padding="max_length",
                            truncation=True, max_length=max_len, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return [id2label[i] for i in preds]


def train_bert(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    model_name: str = "distilbert-base-uncased",
    max_len: int = 128,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 2e-5,
    patience: int = 3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    label_list = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    labels_array = np.array([label2id[l] for l in train_labels])

    class_weights = compute_class_weight("balanced", classes=np.arange(len(label_list)), y=labels_array)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(train_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels_array))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)

    total_steps = len(loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

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

        preds_val = predict_bert(model, tokenizer, id2label, val_texts)
        val_f1 = f1_score(val_labels, preds_val, average="macro", zero_division=0)
        print(f"epoch {epoch+1}/{epochs}  loss={total/len(loader):.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch+1}, best val_f1={best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return model, tokenizer, id2label
