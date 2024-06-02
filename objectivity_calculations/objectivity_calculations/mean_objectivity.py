import pandas as pd
import numpy as np

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_mean_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:

    if len(other_iterator_df) >= options.minimum_number_of_comparison_data:

        mean_objectivity_score = score_calculation(
            same_iterator_df, other_iterator_df, options)

    # Wenn nun aber im Same Iterator und im Other Iterator zusammen über 5 andere Vergleichsdaten vorliegen, dann soll der Faktor angewandt und die Berechnung dennoch durchgeführt werden.
    # Die + 1 folgt aus der ursprünglichen Zeitreihe, die als Grundlage dient und zu diesem Zeitpunkt im same_iterator_df liegt.
    elif len(other_iterator_df) + len(same_iterator_df) >= options.minimum_number_of_comparison_data + 1:
        same_iterator_df_reduced = same_iterator_df.iloc[0:1]
        same_iterator_df_remaining = same_iterator_df.iloc[1:]
        other_iterator_df_expanded = pd.concat(
            [other_iterator_df, same_iterator_df_remaining])

        mean_objectivity_score = score_calculation(
            same_iterator_df_reduced, other_iterator_df_expanded, options, multiple_datasets=False)

    else:
        mean_objectivity_score = 0

    return mean_objectivity_score


def score_calculation(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions, multiple_datasets: bool = True) -> float:
    mean_same_iterator = same_iterator_df.mean(axis=0)
    mean_other_iterator = other_iterator_df.mean(axis=0)

    std_other_iterator = other_iterator_df.std()

    z_score = mean_same_iterator.sub(mean_other_iterator)
    z_score = z_score.where(std_other_iterator != 0, 0).div(
        std_other_iterator.where(std_other_iterator != 0, np.nan)).fillna(0).abs()

    z_score = z_score / options.confidence_interval_z_value

    mean_objectivity_score = 1 - \
        (z_score.mean()**2) / (options.mapping_factor + z_score.mean()**2)

    if multiple_datasets:
        multiple_datasets_factor = 1
    else:
        multiple_datasets_factor = options.factor_for_inner_dataset_calculation_only

    return mean_objectivity_score * multiple_datasets_factor
