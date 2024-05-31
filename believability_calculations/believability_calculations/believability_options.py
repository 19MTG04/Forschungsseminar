import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class BelievabilitySubcategoryWeights:
    accuracy: float = 1

    # Diese beiden Werte werden in Abhängigkeit der jeweiligen rating_array Längen aktualisiert.
    rational_rules: float = 1
    source_data: float = 1

    # Dieser Wert wird in Abhängigkeit der verwendeten Vergleichsdaten aktualisiert und bleibt nicht 1.
    consistency: float = 1

    def sum_weights(self) -> float:
        return sum(self.__dict__.values())


@dataclass
class BelievabilityOptions:
    ratings_array_rational_rules: np.ndarray
    ratings_array_source_data: np.ndarray

    smoothing_factor_rational_rules: float = 0.1

    source_data_used: bool = False
    smoothing_factor_source_data: float = 0.1

    possible_min_max_range: Optional[Tuple[float, float]] = None

    weights: BelievabilitySubcategoryWeights = field(
        default_factory=BelievabilitySubcategoryWeights)


def determine_ratings_array(
    ratings_array: Optional[np.ndarray], length_series: Optional[int], seed: int
) -> np.ndarray:
    if ratings_array is not None and length_series is not None:
        raise ValueError(
            "Es macht keinen Sinn sowohl ratings_array als auch length_series für die gleiche Kategorie zu belegen. Entweder wird ein Array verarbeitet oder eine Zufallsserie vorgegebener Länge erstellt."
        )

    if ratings_array is not None:
        if ratings_array.max() > 1:
            raise ValueError(
                f"Der Bereich der Bewertungen liegt zwischen 0 und 1, aber der Maximalwert des Ratings ist {
                    ratings_array.max()}"
            )
        if ratings_array.min() < 0:
            raise ValueError(
                f"Der Bereich der Bewertungen liegt zwischen 0 und 1, aber der Minimalwert des Ratings ist {
                    ratings_array.min()}"
            )
    else:
        length_series = length_series if length_series is not None else 100
        ratings_array = create_random_series(
            length_series=length_series, seed=seed)

    return ratings_array


def create_believability_options(
    ratings_array_rational_rules: Optional[np.ndarray] = None,
    length_series_rational_rules: Optional[int] = None,
    ratings_array_source_data: Optional[np.ndarray] = None,
    length_series_source_data: Optional[int] = None
) -> BelievabilityOptions:
    ratings_array_rational_rules = determine_ratings_array(
        ratings_array_rational_rules, length_series_rational_rules, seed=2
    )
    ratings_array_source_data = determine_ratings_array(
        ratings_array_source_data, length_series_source_data, seed=3
    )

    options = BelievabilityOptions(
        ratings_array_rational_rules, ratings_array_source_data
    )

    options.weights.rational_rules = determine_weight(
        len(ratings_array_rational_rules))
    options.weights.source_data = determine_weight(
        len(ratings_array_source_data))
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


def determine_weight(number_comparison_series: int) -> float:
    return 1 - (1 - 0.25) * np.exp(-0.02 * number_comparison_series)
