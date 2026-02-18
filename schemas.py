from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


@dataclass
class ModelResult:
    model: str
    bbox: BBox | None
    confidence: float
    reason: str
    latency_ms: int
    raw: Any = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "bbox": self.bbox.as_dict() if self.bbox else None,
            "confidence": self.confidence,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class ConsensusResult:
    winner: BBox | None
    confidence: float
    agreement: float
    chosen_models: list[str] = field(default_factory=list)
    uncertain_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.as_dict() if self.winner else None,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "chosen_models": self.chosen_models,
            "uncertain_reason": self.uncertain_reason,
        }


@dataclass
class InputFrame:
    source: str
    page: int | None
    width: int
    height: int
    image_bytes: bytes
    offset_x: int = 0
    offset_y: int = 0
