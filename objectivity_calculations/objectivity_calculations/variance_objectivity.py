import pandas as pd

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_variance_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:
    mean_same_iterator = same_iterator_df.mean(axis=0)
    mean_other_iterator = other_iterator_df.mean(axis=0)

    # TODO: Bei wie vielen ist das sinnvoll? Vorher checken!
    std_same_iterator = same_iterator_df.std()
    std_other_iterator = other_iterator_df.std()

    variance_factor = max(std_same_iterator.mean(), std_other_iterator.mean()) / \
        min(std_same_iterator.mean(), std_other_iterator.mean()) - 1

    variance_objectivity_score = 1 - \
        (variance_factor**2) / \
        (options.mapping_factor + variance_factor**2)

    return variance_objectivity_score
