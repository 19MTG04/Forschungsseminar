import pandas as pd

from believability_calculations.believability_options import create_believability_options, BelievabilityOptions, determine_weight
from believability_calculations.believability_helper import determine_rational_rules_score, determine_source_data_score, determine_consistency_score

from accuracy_calculations.main_accuracy_score import determine_accuracy
from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions


def calculate_believability_score(accuracy_score: float, data_series: pd.Series, comparison_data: pd.DataFrame, believability_options: BelievabilityOptions) -> float:

    rational_rules_score = determine_rational_rules_score(
        believability_options)

    if believability_options.source_data_used:
        source_data_score = determine_source_data_score(believability_options)
    else:
        source_data_score = 0
        believability_options.weights.source_data = 0

    if len(comparison_data) >= 1:
        consistency_score = determine_consistency_score(
            data_series, comparison_data, believability_options)
    else:
        consistency_score = 0

    believability_options.weights.consistency = determine_weight(
        len(comparison_data))

    believability_score = ((accuracy_score * believability_options.weights.accuracy) +
                           (rational_rules_score * believability_options.weights.rational_rules) +
                           (source_data_score * believability_options.weights.source_data) +
                           (consistency_score * believability_options.weights.consistency)) / (believability_options.weights.sum_weights())
    return believability_score


if __name__ == '__main__':
    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"

    believability_options = create_believability_options()

    accuracy_options = AccuracyCalculationOptions()
    comparison_data_options = ComparisonDataExtractionOptions(
        period_limitations_same_dataset=[('02.10.2023 10:18:00', '02.10.2023 10:20:00'), ('06.10.2023 11:16:00', '06.10.2023 11:18:00')], period_limitations_additional_dataset=[('07.10.2023 11:16:00', '07.10.2023 19:16:00')])

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    accuracy_score = determine_accuracy(
        data_series, accuracy_options, comparison_data)

    believability_score = calculate_believability_score(
        accuracy_score, data_series, comparison_data, believability_options)
    print(believability_score)
