import pandas as pd

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_autocorrelation_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:

    # Die Transponierungen gelten nur dem richtigen axis Zugriff, da der direkte Zugriff deprecated ist
    same_iterator_autocorrelation_sum = same_iterator_df.T.rolling(
        window=options.autocorrlation_horizon).apply(sum_deviations, raw=True).T
    other_iterator_autocorrelation_sum = other_iterator_df.T.rolling(
        window=options.autocorrlation_horizon).apply(sum_deviations, raw=True).T

    mean_same_iterator_ac = same_iterator_autocorrelation_sum.mean(axis=0)
    mean_other_iterator_ac = other_iterator_autocorrelation_sum.mean(axis=0)

    autocorrelation_factor = max(mean_same_iterator_ac.mean(), mean_other_iterator_ac.mean()) / \
        min(mean_same_iterator_ac.mean(), mean_other_iterator_ac.mean()) - 1

    ac_objectivity_score = 1 - \
        (autocorrelation_factor**2) / \
        (options.mapping_factor + autocorrelation_factor**2)

    return ac_objectivity_score


def sum_deviations(window):
    # Abweichung von jedem Wert im Fenster zum ersten Wert, aufsummiert.
    start_value = window[0]
    deviations = (window - start_value) / (len(window) - 1)
    return deviations.sum()
