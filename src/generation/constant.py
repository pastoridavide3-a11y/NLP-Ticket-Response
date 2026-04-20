"""Worst-case-floor constant baseline.

Returns a single polite generic response for every input. Useful as a lower
bound: any real model should beat this on targeted metrics.
"""
from dataclasses import dataclass
from typing import Iterable, List


DEFAULT_RESPONSE = (
    "Thank you for contacting customer support. We have received your message "
    "and a member of our team will review the details and respond as soon as "
    "possible. We appreciate your patience."
)


@dataclass
class ConstantResponder:
    response: str = DEFAULT_RESPONSE

    def fit(self, tickets: Iterable[str], answers: Iterable[str]) -> "ConstantResponder":
        return self

    def predict(self, tickets: Iterable[str]) -> List[str]:
        return [self.response for _ in tickets]
