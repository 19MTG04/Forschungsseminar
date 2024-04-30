import numpy as np
import pandas as pd
from typing import Tuple

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions


def detect_missing_data(data_series: pd.Series, options: AccuracyCalculationOptions) -> Tuple[float, float, int]:
    # Zähle die NaN-Werte
    nan_count = data_series.isna().sum()

    # Berechne den Anteil der NaN-Werte
    nan_rate = nan_count / len(data_series)

    nan_score = np.clip(
        1 - options.multiplicator_missing_data * nan_rate, 0, 1)

    return nan_score, nan_rate, nan_count
