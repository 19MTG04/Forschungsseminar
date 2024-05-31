import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReputationOptions:
    reputation_rating: np.ndarray

    exponential_smoothing_factor: float = 0.1


def create_reputation_options(ratings_array: Optional[np.ndarray] = None, length_series: Optional[int] = None) -> ReputationOptions:
    """
    Erzeugt ein ReputationOptions-Objekt entweder basierend auf einem gegebenen Ratings-Array
    oder durch Generierung einer zufälligen Serie.

    Args:
        ratings_array (Optional[np.ndarray]): Ein Array von Bewertungen, falls vorhanden.
        length_series (Optional[int]): Die Länge der zu generierenden Serie, falls kein ratings_array gegeben ist.

    Returns:
        ReputationOptions: Ein Objekt mit dem Ratings-Array und den anderen Klassenwerten.
    """
    if ratings_array is not None and length_series is not None:
        raise ValueError(
            "Es macht keinen Sinn sowohl ratings_array als auch length_series zu belegen. Entweder wird ein Array verarbeitet oder eine Zufallsserie vorgegebener Länge erstellt.")

    if ratings_array is not None:
        pass
    else:
        length_series = length_series if length_series is not None else 100
        ratings_array = create_random_series(
            length_series=length_series, seed=1)

    options = ReputationOptions(ratings_array)
    return options


def create_random_series(length_series: int, seed: int = 1) -> np.ndarray:
    # Reproduzierbare Ergebnisse mit Zufallszahlen immer über seed steuern
    np.random.seed(seed)

    # Trend nachbilden und mit Zufallszahlen abändern
    trend = np.array([0 + (i**2) * 0.5 / length_series **
                     2 for i in range(length_series)])
    random_values = np.random.uniform(low=-0.2, high=0.2, size=length_series)
    ratings_array = np.clip(trend + random_values, 0, 1)

    return ratings_array
