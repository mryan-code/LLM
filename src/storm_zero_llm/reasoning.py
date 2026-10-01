"""Small response reasoning layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningResult:
    response: str
    confidence: str
    critique: str


class ReasoningEngine:
    def wrap(self, response: str, used_memory: bool = False) -> ReasoningResult:
        lowered = response.lower()
        if any(term in lowered for term in ("not sure", "uncertain", "i do not know", "i don't know")):
            confidence = "uncertain"
            critique = "Response communicates uncertainty."
        else:
            confidence = "normal"
            critique = "Response generated from conversation context."
        return ReasoningResult(response=response, confidence=confidence, critique=critique)
