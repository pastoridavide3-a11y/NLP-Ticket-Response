"""Subgroup-conditioned template generator.

For each (priority, department) pair, pick the centroid-nearest train answer
as the group's fixed response. A label-conditioned baseline that ties to the
optional mitigation question: does conditioning on subgroup labels close gaps?
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


GroupKey = Tuple[str, str]


@dataclass
class GroupTemplateResponder:
    fallback: str = (
        "Thank you for reaching out. We have received your request and will get "
        "back to you with more details shortly."
    )
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000
        )
    )
    _templates: Dict[GroupKey, str] = field(default_factory=dict, init=False)

    def fit(
        self,
        answers: Sequence[str],
        priorities: Sequence[str],
        departments: Sequence[str],
    ) -> "GroupTemplateResponder":
        answers = list(answers)
        X = self.vectorizer.fit_transform(answers)
        groups = list(zip(priorities, departments))
        by_group: Dict[GroupKey, List[int]] = {}
        for i, g in enumerate(groups):
            by_group.setdefault(g, []).append(i)
        for g, idx in by_group.items():
            sub = X[idx]
            centroid = sub.mean(axis=0)
            centroid = np.asarray(centroid)
            # cosine to centroid: normalized dot
            sub_norm = sub.multiply(1.0 / (np.sqrt(sub.multiply(sub).sum(axis=1)) + 1e-9))
            cen_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            sims = sub_norm @ cen_norm.T
            sims = np.asarray(sims).ravel()
            best_local = int(np.argmax(sims))
            self._templates[g] = answers[idx[best_local]]
        return self

    def predict(
        self, priorities: Sequence[str], departments: Sequence[str]
    ) -> List[str]:
        out = []
        for p, d in zip(priorities, departments):
            out.append(self._templates.get((p, d), self.fallback))
        return out
