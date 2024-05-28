from dataclasses import dataclass, field
from typing import List


@dataclass
class ReputationOptions:
    exponential_smoothing_factor: float = 0.1

    reputation_rating: List[float] = field(default_factory=list)
