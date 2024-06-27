import pandas as pd
import numpy as np

from believability_calculations.believability_options import BelievabilityOptions

from base_library.community_score import calculate_community_score


def determine_rational_rules_score(options: BelievabilityOptions) -> float:
    rational_rules_score = calculate_community_score(
        options.ratings_array_rational_rules, options.smoothing_factor_rational_rules)

    return rational_rules_score


def determine_source_data_score(options: BelievabilityOptions) -> float:
    source_data_score = calculate_community_score(
        options.ratings_array_source_data, options.smoothing_factor_source_data)

    return source_data_score


def determine_consistency_score(data_series: pd.Series, comparison_data: pd.DataFrame, options: BelievabilityOptions) -> float:
    if options.possible_min_max_range is not None:
        potential_minimum = options.possible_min_max_range[0]
        potential_maximum = options.possible_min_max_range[1]

        if potential_maximum <= potential_minimum:
            raise ValueError(
                f"Die vorgegebene Range für realistische Daten ist fehlerhaft. {potential_minimum=}, {potential_maximum=}")
    else:
        potential_minimum = comparison_data.min(axis=None)
        potential_maximum = comparison_data.max(axis=None)

    comparison_mean = comparison_data.mean(
        axis=0).apply(pd.to_numeric).fillna(0).values

    valid_indices = ~data_series.isna()

    consistency_score = (1 - np.abs(np.array(data_series[valid_indices].values) - comparison_mean[valid_indices]) /
                         (potential_maximum - potential_minimum)).mean()

    return consistency_score
