import pandas as pd

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_variance_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:

    if len(same_iterator_df) >= 5 and len(other_iterator_df) >= 5:
        std_same_iterator = same_iterator_df.std()
        std_other_iterator = other_iterator_df.std()

        variance_factor = max(std_same_iterator.mean(), std_other_iterator.mean()) / \
            min(std_same_iterator.mean(), std_other_iterator.mean()) - 1

        variance_objectivity_score = 1 - \
            (variance_factor**2) / \
            (options.mapping_factor + variance_factor**2)
    else:
        variance_objectivity_score = 0

    return variance_objectivity_score
