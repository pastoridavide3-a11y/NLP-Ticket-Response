"""Lightweight heuristic failure tagger.

Automatic pre-labels every prediction with one or more heuristic tags so that
the manual review (Phase 7) only needs to confirm / refine rather than
originate the category. Tags are flags, not mutually exclusive.
"""
from dataclasses import dataclass
from typing import Dict, List


GENERIC_CUES = (
    "thank you for contacting",
    "we appreciate your patience",
    "get back to you",
    "as soon as possible",
    "review your request",
)


@dataclass
class HeuristicTagger:
    short_char_threshold: int = 80
    long_char_multiplier: float = 2.5
    low_overlap_ratio: float = 0.15

    def tag(self, pred: str, gold: str) -> Dict[str, bool]:
        pred = pred or ""
        gold = gold or ""
        tags: Dict[str, bool] = {}
        tags["too_short"] = len(pred) < self.short_char_threshold
        tags["too_long"] = len(pred) > self.long_char_multiplier * max(len(gold), 1)
        tags["generic"] = any(cue in pred.lower() for cue in GENERIC_CUES)

        pred_toks = set(pred.lower().split())
        gold_toks = set(gold.lower().split())
        if gold_toks:
            overlap = len(pred_toks & gold_toks) / max(len(gold_toks), 1)
        else:
            overlap = 0.0
        tags["low_overlap"] = overlap < self.low_overlap_ratio
        return tags

    def tag_batch(self, preds: List[str], golds: List[str]) -> List[Dict[str, bool]]:
        return [self.tag(p, g) for p, g in zip(preds, golds)]
