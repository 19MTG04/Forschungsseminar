import pandas as pd

from data_quality_calculations.data_quality_options import CategoryWeightsDataQuality, ModelType, DataQualityOptions
from base_library.data_extraction_options import ComparisonDataExtractionOptions
from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.main_accuracy_score import determine_accuracy

from believability_calculations.believability_options import BelievabilityOptions, create_believability_options
from believability_calculations.main_believability_calculation import calculate_believability_score

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions, create_objectivity_calculation_options
from objectivity_calculations.objectivity_calculation import calculate_obejectivity

from reputation_calculations.reputation_options import ReputationOptions, create_reputation_options
from reputation_calculations.reputation_calculation import calculate_reputation_score


def calculate_final_score(model_type: ModelType, data_series: pd.Series, comparison_data: pd.DataFrame, comparison_data_weights: CategoryWeightsDataQuality, comparison_data_options: ComparisonDataExtractionOptions, data_quality_options: DataQualityOptions) -> float:
    accuracy_score = determine_accuracy(
        data_series, data_quality_options.accuracy_options, comparison_data)
    believability_score = calculate_believability_score(
        accuracy_score, data_series, comparison_data, data_quality_options.believability_options)
    objectivity_score = calculate_obejectivity(
        data_series, comparison_data, comparison_data_options, data_quality_options.objectivity_options)
    reputation_score = calculate_reputation_score(
        data_quality_options.reputation_options.reputation_rating, data_quality_options.reputation_options.exponential_smoothing_factor)

    # TODO: Main fertig machen
    # TODO: Prüfen, ob die Aufrufe so stimmen
    # TODO: Berechnung des Gesamtscoers einfügen
    # TODO: ModelType mit einbeziehen
    # TODO: Ausgiebig testen!
    return 1


if __name__ == '__main__':

    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"

    model_type = ModelType.FULL_MODEL

    comparison_data_options = ComparisonDataExtractionOptions(
        period_limitations_same_dataset=[('02.10.2023 10:18:00', '02.10.2023 10:20:00'), ('06.10.2023 11:16:00', '06.10.2023 11:18:00')], period_limitations_additional_dataset=[('07.10.2023 11:16:00', '07.10.2023 19:16:00')])

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    accuracy_options = AccuracyCalculationOptions()
    believability_options = create_believability_options()
    objectivity_options = create_objectivity_calculation_options(data_series)
    reputation_options = create_reputation_options()
    data_quality_options = DataQualityOptions(
        accuracy_options, believability_options, objectivity_options, reputation_options)

    data_quality_score = calculate_final_score()

    print(f'Datenqualitäts-Score: {data_quality_score:.2f}')
