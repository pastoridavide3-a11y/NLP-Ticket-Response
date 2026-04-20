"""Semantic-similarity proxies using TF-IDF cosine.

Without a transformer available in the base environment, we use TF-IDF word
+ char n-gram cosine as a tractable proxy for semantic similarity. This is a
weaker signal than BERTScore but still captures lexical-topical overlap
beyond single-n-gram ROUGE. Documented in EVALUATION.md as Phase-1 semantic
evidence; BERTScore is listed as a Phase-2 upgrade in the same file.
"""
from typing import Dict, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_cosine(preds: Sequence[str], refs: Sequence[str]) -> np.ndarray:
    """Pairwise cosine between each prediction and its reference (shape [N])."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    corpus = list(preds) + list(refs)
    X = vec.fit_transform(corpus)
    n = len(preds)
    P, R = X[:n], X[n:]
    # Row-wise cosine
    def row_norm(M):
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) + 1e-12
        return M.multiply(1.0 / norms[:, None])

    P = row_norm(P).tocsr()
    R = row_norm(R).tocsr()
    sims = np.asarray(P.multiply(R).sum(axis=1)).ravel()
    return sims


def length_ratio(preds: Sequence[str], refs: Sequence[str]) -> np.ndarray:
    """Character-length ratio pred/ref. 1.0 = match; <1 short; >1 long."""
    lp = np.array([len(p or "") for p in preds], dtype=float)
    lr = np.array([len(r or "") for r in refs], dtype=float)
    return lp / np.maximum(lr, 1.0)


def all_semantic(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, np.ndarray]:
    return {
        "tfidf_cos": tfidf_cosine(preds, refs),
        "len_ratio": length_ratio(preds, refs),
    }
