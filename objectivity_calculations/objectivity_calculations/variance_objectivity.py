import pandas as pd
import numpy as np

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions


def calculate_variance_objectivity(same_iterator_df: pd.DataFrame, other_iterator_df: pd.DataFrame, options: ObjectivityCalculationOptions) -> float:

    if len(same_iterator_df) >= options.minimum_number_of_comparison_data and len(other_iterator_df) >= options.minimum_number_of_comparison_data:
        std_same_iterator = same_iterator_df.std()
        std_other_iterator = other_iterator_df.std()

        # In jedem Zeitschritt wird der größere druch den kleineren Wert geteilt. Hier ist dies unproblematisch, da jeder der Werte immer >= 0 sein wird.
        variance_factor = std_same_iterator.combine(
            std_other_iterator, lambda x, y: (max(x, y) / min(x, y) - 1) if min(x, y) != 0 else np.nan)

        variance_objectivity_score = 1 - \
            (variance_factor.mean()**2) / \
            (options.mapping_factor + variance_factor.mean()**2)
    else:
        variance_objectivity_score = 0

    return variance_objectivity_score
