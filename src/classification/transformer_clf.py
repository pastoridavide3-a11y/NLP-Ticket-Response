"""DistilBERT fine-tuning helpers — plain functions, no classes."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def tokenize(texts: list[str], tokenizer, max_len: int = 128) -> dict:
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")


def train_bert(
    train_texts: list[str],
    train_labels: list[str],
    model_name: str = "distilbert-base-uncased",
    max_len: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_list = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    int_labels = [label2id[l] for l in train_labels]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenize(train_texts, tokenizer, max_len)
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(int_labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return model, tokenizer, id2label


def predict_bert(model, tokenizer, id2label: dict, texts: list[str], max_len: int = 128, batch_size: int = 64) -> list[str]:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return [id2label[i] for i in preds]
