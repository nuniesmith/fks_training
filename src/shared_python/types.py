from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class BaselineModelParams:
    model_type: str
    max_depth: int | None = None
    learning_rate: float | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def effective_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model_type": self.model_type}
        if self.max_depth is not None:
            out["max_depth"] = self.max_depth
        if self.learning_rate is not None:
            out["learning_rate"] = self.learning_rate
        out.update(self.extra)
        return out

__all__ = ["BaselineModelParams"]
