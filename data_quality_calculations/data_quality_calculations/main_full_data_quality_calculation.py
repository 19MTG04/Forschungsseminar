import pandas as pd
from typing import Tuple, Union

from data_quality_calculations.data_quality_options import CategoryWeightsDataQuality, ModelType, DataQualityOptions
from base_library.data_extraction_options import ComparisonDataExtractionOptions
from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.main_accuracy_score import determine_accuracy

from believability_calculations.believability_options import create_believability_options
from believability_calculations.main_believability_calculation import calculate_believability_score

from objectivity_calculations.objectivity_options import create_objectivity_calculation_options
from objectivity_calculations.objectivity_calculation import calculate_obejectivity

from reputation_calculations.reputation_options import create_reputation_options
from reputation_calculations.reputation_calculation import calculate_reputation_score


def calculate_final_score(model_type: ModelType, data_series: pd.Series, comparison_data: pd.DataFrame, data_quality_weights: CategoryWeightsDataQuality, comparison_data_options: ComparisonDataExtractionOptions, data_quality_options: DataQualityOptions) -> Tuple[float, Tuple[float, float, float, Union[float, str]]]:
    if model_type == ModelType.FULL_MODEL:
        accuracy_score = determine_accuracy(
            data_series, data_quality_options.accuracy_options, comparison_data)
        believability_score = calculate_believability_score(
            accuracy_score, data_series, comparison_data, data_quality_options.believability_options)
        objectivity_score = calculate_obejectivity(
            data_series, comparison_data, comparison_data_options, data_quality_options.objectivity_options)
        reputation_score = calculate_reputation_score(
            data_quality_options.reputation_options.reputation_rating, data_quality_options.reputation_options.exponential_smoothing_factor)

        data_quality_score = ((accuracy_score * data_quality_weights.weight_accuracy) +
                              (believability_score * data_quality_weights.weight_believability) +
                              (objectivity_score * data_quality_weights.weight_objectivity) +
                              (reputation_score * data_quality_weights.weight_reputation)) / (data_quality_weights.sum_weights())

    elif model_type == ModelType.MODEL_WO_COMMUNITY:
        data_quality_options.believability_options.weights.rational_rules = 0
        data_quality_options.believability_options.weights.source_data = 0
        data_quality_weights.weight_reputation = 0

        accuracy_score = determine_accuracy(
            data_series, data_quality_options.accuracy_options, comparison_data)
        believability_score = calculate_believability_score(
            accuracy_score, data_series, comparison_data, data_quality_options.believability_options)
        objectivity_score = calculate_obejectivity(
            data_series, comparison_data, comparison_data_options, data_quality_options.objectivity_options)
        reputation_score = "Keine Berechnung für diesen ModelType."

        data_quality_score = ((accuracy_score * data_quality_weights.weight_accuracy) +
                              (believability_score * data_quality_weights.weight_believability) +
                              (objectivity_score * data_quality_weights.weight_objectivity)) / (data_quality_weights.sum_weights())

    else:
        raise ValueError(f"Der Typ {model_type=} ist unbekannt.")

    return data_quality_score, (accuracy_score, believability_score, objectivity_score, reputation_score)


if __name__ == '__main__':

    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"

    model_type = ModelType.MODEL_WO_COMMUNITY

    comparison_data_options = ComparisonDataExtractionOptions(
        minimum_comparison_data_duration_sec=100,
        period_limitations_same_dataset=[],
        period_limitations_additional_dataset=[])

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    accuracy_options = AccuracyCalculationOptions(
        plot_intrinsic_outliers=True, plot_comparison_data=True)
    believability_options = create_believability_options()
    objectivity_options = create_objectivity_calculation_options(data_series)
    reputation_options = create_reputation_options()
    data_quality_options = DataQualityOptions(
        accuracy_options, believability_options, objectivity_options, reputation_options)
    data_quality_weights = CategoryWeightsDataQuality()

    data_quality_score, category_scores = calculate_final_score(
        model_type, data_series, comparison_data, data_quality_weights, comparison_data_options, data_quality_options)

    if isinstance(category_scores[3], str):
        category_score_3 = category_scores[3]
    else:
        category_score_3 = f'{category_scores[3]:.2f}'

    print(f'Datenqualitäts-Score: {data_quality_score:.2f}\n'
          f'Genauigkeit: {category_scores[0]:.2f},\n'
          f'Glaubwürdigkeit: {category_scores[1]:.2f},\n'
          f'Objektivität: {category_scores[2]:.2f},\n'
          f'Ruf: {category_score_3}')
