import pandas as pd
import numpy as np

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_autocorrelation_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, comparison_data: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:
    # Dient nur dem kürzeren Variablennamen
    horizon = options.autocorrlation_horizon

    # Die Transponierungen gelten nur dem richtigen axis Zugriff, da der direkte Zugriff deprecated ist
    same_iterator_autocorrelation_sum = same_iterator_df.T.rolling(
        window=horizon, min_periods=int(horizon / 2)).apply(sum_deviations, raw=True).T
    other_iterator_autocorrelation_sum = other_iterator_df.T.rolling(
        window=horizon, min_periods=int(horizon / 2)).apply(sum_deviations, raw=True).T

    mean_same_iterator_ac = same_iterator_autocorrelation_sum.mean(axis=0)
    mean_other_iterator_ac = other_iterator_autocorrelation_sum.mean(axis=0)

    # Differenz der beiden Serien und diese dann auf die jeweilige Größenordnung in den Gesamtvergleichsdaten beziehen
    ac_magnitude_series = (abs(mean_same_iterator_ac -
                           mean_other_iterator_ac)) / abs(comparison_data.mean(axis=0))

    # Inklusive Faktor zur Gewichtung, wie viele Daten wirklich zur Verfügung stehen.
    autocorrelation_factor = ac_magnitude_series.mean() * (len(mean_same_iterator_ac) - mean_same_iterator_ac.isna(
    ).sum()) / (len(mean_other_iterator_ac) - mean_other_iterator_ac.isna().sum())

    ac_objectivity_score = 1 - \
        (autocorrelation_factor**2) / \
        (options.mapping_factor + autocorrelation_factor**2)

    return ac_objectivity_score


def sum_deviations(window):
    # Entferne NaN-Werte aus dem Fenster
    valid_window = window[~np.isnan(window)]
    if len(valid_window) == 0:
        return np.nan  # Wenn alle Werte NaN sind, gib NaN zurück

    # Abweichung von jedem Wert im Fenster zum ersten Wert, aufsummiert
    start_value = valid_window[0]
    deviations = (valid_window - start_value) / (len(valid_window) - 1)
    return deviations.sum()
