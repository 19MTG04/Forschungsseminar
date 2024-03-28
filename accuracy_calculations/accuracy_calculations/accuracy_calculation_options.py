from dataclasses import dataclass
from typing import Union, Literal


@dataclass
class AccuracyCalculationOptions:
    mode_for_dispersion_identification: Union[Literal['mean'],
                                              Literal['regression_deg_2'],
                                              Literal['regression_deg_3']] = 'mean'
    plot_outliers: bool = False
