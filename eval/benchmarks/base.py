"""Base classes for benchmark data loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    """A single evaluation sample from a benchmark."""
    sample_id: str
    video_path: str
    question: str
    ground_truth: str
    query_timestamp: float
    options: list[str] = field(default_factory=list)
    correct_option_idx: int = -1
    metadata: dict = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return len(self.options) > 0 and self.correct_option_idx >= 0


class BaseBenchmark(ABC):
    """Abstract benchmark loader."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load_samples(self) -> list[EvalSample]:
        """Load all evaluation samples for this benchmark."""
        ...

    def validate(self) -> bool:
        """Check that required data files exist."""
        if not self.data_root.exists():
            print(f"[{self.name}] Data root not found: {self.data_root}")
            return False
        return True
