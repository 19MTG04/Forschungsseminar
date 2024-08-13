import pandas as pd
import numpy as np
from typing import Tuple

from accuracy_calculations.missing_data import detect_missing_data
from accuracy_calculations.outliers_no_comp import detect_outliers_intrinsic
from accuracy_calculations.dispersion_no_comp import get_dispersion_stats
from accuracy_calculations.statistical_analysis_helper import general_dispersion_analysis
from accuracy_calculations.comparison_data_score_analysis import get_dispersion_and_outlier_score_for_comparison_data
from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions

from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions

# TODO: Ich nutze die Berechnung der Standardabweichung für eine Stichprobe (ddof=1).
# TODO: Im Bericht steht als Konfidenzintervall 95%. Eher 99% nutzen wie hier im Code oder?
# TODO: 2.1.5 ist im Bericht tendenziell falsch mit den Formeln. Pi/2 und besser beschreiben wie es im Code ist.
# TODO: Gesamte Berechnung für intrinsische Genauigkeitsberechnung ist im Bericht noch nicht beschrieben. Im Code sollte alles passen.
# TODO: NaN-Handling ist bisher nicht überall explizit betrachtet worden!


def determine_accuracy(data_series: pd.Series, accuracy_options: AccuracyCalculationOptions, comparison_data: pd.DataFrame = pd.DataFrame(None)) -> Tuple[float, dict]:
    sanity_check_data(data_series)
    accuracy_score, accuracy_subscores = calculate_accuracy_score(
        data_series, accuracy_options, comparison_data)
    return accuracy_score, accuracy_subscores


def calculate_accuracy_score(data_series: pd.Series, accuracy_options: AccuracyCalculationOptions, comparison_data: pd.DataFrame = pd.DataFrame(None)) -> Tuple[float, dict]:

    window_length, approximation_curve, z_score = general_dispersion_analysis(
        data_series, accuracy_options)

    missing_data_score, _, _ = detect_missing_data(
        data_series, accuracy_options)
    outlier_score, _, _, _ = detect_outliers_intrinsic(
        data_series, accuracy_options, approximation_curve, z_score)
    dispersion_score = get_dispersion_stats(
        data_series, approximation_curve, z_score, accuracy_options, window_length)

    if len(comparison_data) >= accuracy_options.minimum_number_of_comparison_data:
        dispersion_score_comparison, outlier_score_comparison, _ = get_dispersion_and_outlier_score_for_comparison_data(
            data_series, comparison_data, accuracy_options)
    else:
        dispersion_score_comparison = 0
        outlier_score_comparison = 0
        accuracy_options.weights.dispersion_comparison = 0
        accuracy_options.weights.outliers_comparison = 0

    accuracy_score = ((missing_data_score * accuracy_options.weights.missing_data) +
                      (outlier_score * accuracy_options.weights.outliers_intrinsic) +
                      (dispersion_score * accuracy_options.weights.dispersion_intrinsic) +
                      (outlier_score_comparison * accuracy_options.weights.outliers_comparison) +
                      (dispersion_score_comparison * accuracy_options.weights.dispersion_comparison)) / (accuracy_options.weights.sum_weights())

    accuracy_subscores = {
        'missing_data_score': missing_data_score,
        'dispersion_score': dispersion_score,
        'outlier_score': outlier_score,
        'dispersion_score_comparison': dispersion_score_comparison,
        'outlier_score_comparison': outlier_score_comparison
    }

    return accuracy_score, accuracy_subscores


def sanity_check_data(data_series: pd.Series) -> None:
    unique_values = data_series.unique()
    if len(unique_values) == 1 and unique_values[0] == 0:
        raise ValueError(
            "Die Zeitreihe besteht ausschließlich aus Nullen. Es wird keine weitere Berechnung durchgeführt.")


if __name__ == '__main__':
    def main():
        N = 1000
        np.random.seed(1)
        df = pd.DataFrame(
            {'test_data': np.sin(np.linspace(0, 10, num=N)) + np.random.normal(scale=0.6, size=N)})
        data_series = df['test_data']

        accuracy_options = AccuracyCalculationOptions(
            plot_intrinsic_outliers=True)

        accuracy_score, _ = calculate_accuracy_score(
            data_series, accuracy_options)

        print(f'Genauigkeits-Score: {accuracy_score:.2f}')

    # main()

    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"
    accuracy_options = AccuracyCalculationOptions(
        plot_intrinsic_outliers=True, plot_comparison_data=True)
    # comparison_data_options = ComparisonDataExtractionOptions()
    comparison_data_options = ComparisonDataExtractionOptions(
        period_limitations_same_dataset=[('02.10.2023 10:18:00', '02.10.2023 10:20:00'), ('06.10.2023 11:16:00', '06.10.2023 11:18:00')], period_limitations_additional_dataset=[('07.10.2023 11:16:00', '07.10.2023 19:16:00')])

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    score, _ = determine_accuracy(
        data_series, accuracy_options, comparison_data)

    print(f'Genauigkeits-Score: {score:.2f}')
