import pandas as pd
from typing import Tuple


def detect_missing_data(data_series: pd.Series) -> Tuple[float, int]:
    # Zähle die NaN-Werte
    nan_count = data_series.isna().sum()

    # Berechne den Anteil der NaN-Werte
    nan_rate = nan_count / len(data_series)

    return nan_rate, nan_count
