import pandas as pd

from objectivity_calculations.objectivity_helper import extract_dataframes_for_objectivity_calculations, calculate_mean_obejectivity, calculate_variance_obejectivity, calculate_autocorrelation_obejectivity
from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions, create_objectivity_calculation_options

from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions


def calculate_obejectivity(data_series: pd.Series, comparison_data: pd.DataFrame, comparison_data_options: ComparisonDataExtractionOptions, objectivity_options: ObjectivityCalculationOptions) -> float:
    same_iterator_df, other_iterator_df, one_dataset_only = extract_dataframes_for_objectivity_calculations(
        data_series, comparison_data, comparison_data_options)

    # Wenn keine Vergleichsdaten vorhanden sind, dann ist der Score automatisch 0
    if len(other_iterator_df) == 0 and one_dataset_only:
        objectivity_score = 0
    else:
        # Ansonsten kann die Berechnung ganz normal geschehen. Zum Abschluss muss jedoch berücksichtigt werden, ob einer oder mehrere Datensätze die Grundlage waren.
        if one_dataset_only:
            dataset_factor = objectivity_options.factor_for_inner_dataset_calculation_only
        else:
            dataset_factor = 1

        mean_objectivity_score = calculate_mean_obejectivity(
            same_iterator_df, other_iterator_df)
        variance_objectivity_score = calculate_variance_obejectivity(
            same_iterator_df, other_iterator_df)
        autocorrelation_objectivity_score = calculate_autocorrelation_obejectivity(
            same_iterator_df, other_iterator_df)

        objectivity_score = dataset_factor * ((mean_objectivity_score * objectivity_options.weights.mean_objectivity) +
                                              (variance_objectivity_score * objectivity_options.weights.variance_objectivity) +
                                              (autocorrelation_objectivity_score *
                                               objectivity_options.weights.autocorrelation_objectivity)
                                              ) / (objectivity_options.weights.sum_weights())

    return objectivity_score


if __name__ == '__main__':
    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"

    comparison_data_options = ComparisonDataExtractionOptions(
        period_limitations_same_dataset=[('02.10.2023 10:18:00', '02.10.2023 10:20:00'), ('05.10.2023 11:16:00', '06.10.2023 11:18:00')], period_limitations_additional_dataset=[('07.10.2023 11:16:00', '07.10.2023 19:16:00'), ('08.10.2023 11:16:00', '09.10.2023 19:16:00')])

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    objectivity_options = create_objectivity_calculation_options(data_series)

    objectivity_score = calculate_obejectivity(data_series, comparison_data,
                                               comparison_data_options, objectivity_options)
    print(objectivity_score)
