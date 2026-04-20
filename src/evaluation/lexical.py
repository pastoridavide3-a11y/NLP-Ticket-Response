"""Per-example lexical metrics: ROUGE-1/2/L, BLEU, chrF.

All metrics are computed per-example to enable subgroup aggregation and
bootstrap confidence intervals downstream.
"""
from typing import Dict, List, Sequence
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_row(pred: str, ref: str) -> Dict[str, float]:
    pred = pred or ""
    ref = ref or ""
    r = _ROUGE.score(ref, pred)
    bleu = sacrebleu.sentence_bleu(pred, [ref]).score  # 0..100
    chrf = sacrebleu.sentence_chrf(pred, [ref]).score  # 0..100
    return {
        "rouge1": r["rouge1"].fmeasure,
        "rouge2": r["rouge2"].fmeasure,
        "rougeL": r["rougeL"].fmeasure,
        "bleu": bleu / 100.0,
        "chrf": chrf / 100.0,
    }


def score_batch(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, float]] = [score_row(p, r) for p, r in zip(preds, refs)]
    keys = ("rouge1", "rouge2", "rougeL", "bleu", "chrf")
    return {k: np.asarray([row[k] for row in rows], dtype=float) for k in keys}
