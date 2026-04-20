"""Nearest-neighbor retrieval generator over TF-IDF vectors.

Given a ticket, encode with TF-IDF, find the most similar train ticket, and
return that ticket's gold answer. Serves as the primary non-trivial baseline.
"""
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

from src.features.tfidf import build_tfidf


@dataclass
class TfidfRetriever:
    k: int = 1
    featurizer: Optional[object] = None
    _train_answers: List[str] = field(default_factory=list, init=False)
    _train_matrix: Optional[sparse.spmatrix] = field(default=None, init=False)

    def fit(self, tickets: Sequence[str], answers: Sequence[str]) -> "TfidfRetriever":
        if self.featurizer is None:
            self.featurizer = build_tfidf()
        self._train_matrix = self.featurizer.fit_transform(list(tickets))
        self._train_answers = list(answers)
        return self

    def predict(self, tickets: Iterable[str], batch_size: int = 512) -> List[str]:
        tickets = list(tickets)
        X = self.featurizer.transform(tickets)
        preds: List[str] = []
        for i in range(0, X.shape[0], batch_size):
            block = X[i : i + batch_size]
            sims = linear_kernel(block, self._train_matrix)
            idx = np.argmax(sims, axis=1)
            preds.extend(self._train_answers[j] for j in idx)
        return preds

    def predict_topk(self, tickets: Iterable[str], k: int, batch_size: int = 512):
        """Return top-k neighbor indices & scores. Used by diagnostic scripts."""
        tickets = list(tickets)
        X = self.featurizer.transform(tickets)
        all_idx, all_sim = [], []
        for i in range(0, X.shape[0], batch_size):
            block = X[i : i + batch_size]
            sims = linear_kernel(block, self._train_matrix)
            part = np.argsort(-sims, axis=1)[:, :k]
            all_idx.append(part)
            all_sim.append(np.take_along_axis(sims, part, axis=1))
        return np.vstack(all_idx), np.vstack(all_sim)
