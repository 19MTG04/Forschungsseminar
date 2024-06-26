import pandas as pd

from dataclasses import dataclass, field
from typing import Union, Literal


@dataclass
class ObjectivitySubcategoryWeights:
    mean_objectivity: float = 1
    variance_objectivity: float = 1
    autocorrelation_objectivity: float = 1

    def sum_weights(self) -> float:
        return sum(self.__dict__.values())


@dataclass
class ObjectivityCalculationOptions():
    autocorrlation_horizon: int

    factor_for_inner_dataset_calculation_only: float = 0.8

    minimum_number_of_comparison_data: int = 5

    mapping_factor: float = 0.5

    confidence_interval_z_value: float = 2.58  # 99% Konfidenzintervall

    weights: ObjectivitySubcategoryWeights = field(
        default_factory=ObjectivitySubcategoryWeights)


def create_objectivity_calculation_options(series: pd.Series):
    return ObjectivityCalculationOptions(autocorrlation_horizon=max(5, int(len(series) / 10)))
