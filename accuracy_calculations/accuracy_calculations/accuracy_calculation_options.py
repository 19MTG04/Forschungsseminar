from dataclasses import dataclass, field
from typing import Union, Literal


@dataclass
class AccuracySubcategoryWeights:
    missing_data: float = 1
    outliers_intrinsic: float = 1
    dispersion_intrinsic: float = 1

    outliers_comparison: float = 2
    dispersion_comparison: float = 2

    def sum_weights(self) -> float:
        return sum(self.__dict__.values())


@dataclass
class AccuracyCalculationOptions:
    mode_for_dispersion_identification: Union[Literal['mean'],
                                              Literal['regression_deg_1'],
                                              Literal['regression_deg_2'],
                                              Literal['regression_deg_3']] = 'regression_deg_1'
    # Aufgrund des Rechenaufwandes wird hier der Durchschnitt als Default gesetzt.
    mode_for_window_length_identification: Union[Literal['mean'],
                                                 Literal['regression_deg_1'],
                                                 Literal['regression_deg_2'],
                                                 Literal['regression_deg_3']] = 'mean'
    threshold_outliers: float = 2.58

    plot_intrinsic_outliers: bool = False

    weights: AccuracySubcategoryWeights = field(
        default_factory=AccuracySubcategoryWeights)
