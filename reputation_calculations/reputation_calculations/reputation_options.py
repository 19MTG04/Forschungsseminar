import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReputationOptions:
    reputation_rating: np.ndarray

    exponential_smoothing_factor: float = 0.1


def create_reputation_options(ratings_array: Optional[np.ndarray] = None, length_series: int = 100) -> ReputationOptions:
    if ratings_array is not None:
        pass
    else:
        ratings_array = create_random_series(length_series=length_series)

    options = ReputationOptions(ratings_array)
    return options


def create_random_series(length_series: int) -> np.ndarray:
    # Reproduzierbare Ergebnisse mit Zufallszahlen immer über seed steuern
    np.random.seed(1)

    # Trend nachbilden und mit Zufallszahlen abändern
    trend = [0 + (i**2) * 0.5 / length_series**2 for i in range(length_series)]
    ratings_array = np.add(np.random.uniform(
        low=-0.2, high=0.2, size=length_series), trend)

    ratings_array = np.clip(ratings_array, 0, 1)

    return ratings_array
