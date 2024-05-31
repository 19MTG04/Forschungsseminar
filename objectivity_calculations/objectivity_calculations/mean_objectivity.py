import pandas as pd

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_mean_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:
    mean_same_iterator = same_iterator_df.mean(axis=0)
    mean_other_iterator = other_iterator_df.mean(axis=0)

    if len(other_iterator_df) >= options.minimum_number_of_comparison_data:
        std_other_iterator = other_iterator_df.std()

        z_score = mean_same_iterator.sub(
            mean_other_iterator).div(std_other_iterator).fillna(0).abs()
        z_score = z_score / options.confidence_interval_z_value

        mean_objectivity_score = 1 - \
            (z_score.mean()**2) / (options.mapping_factor + z_score.mean()**2)
    else:
        mean_objectivity_score = 0

    return mean_objectivity_score
