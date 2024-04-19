import pandas as pd
import numpy as np
import tdm_loader
from pathlib import Path

from accuracy_calculations.missing_data import detect_missing_data
from accuracy_calculations.outliers_no_comp import detect_outliers_intrinsic
from accuracy_calculations.dispersion_no_comp import get_dispersion_stats
from accuracy_calculations.statistical_analysis_helper import general_dispersion_analysis
from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.misc.path_helper import get_project_root

from base_library.extract_data_and_comparison_data import extract_relevant_data, extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions

# TODO: Ich nutze nicht die Berechnung der Standardabweichung für eine Stichprobe bisher. Sollte ich? (ddof=0 im Moment, bei Testdaten besser)
# TODO: Im Bericht steht als Konfidenzintervall 95%. Eher 99% nutzen wie hier im Code oder?
# TODO: 2.1.5 ist im Bericht tendenziell falsch mit den Formeln. Pi/2 und besser beschreiben wie es im Code ist.
# TODO: Gesamte Berechnung für intrinsische Genauigkeitsberechnung ist im Bericht noch nicht beschrieben. Im Code sollte alles passen.
# TODO: NaN-Handling ist bisher nicht überall explizit betrachtet worden!


def determine_accuracy(data_series: pd.Series, accuracy_options: AccuracyCalculationOptions, comparison_data: pd.DataFrame = pd.DataFrame(None)) -> float:
    sanity_check_data(data_series)
    accuracy_score = calculate_accuracy_score(
        data_series, accuracy_options, comparison_data)
    return accuracy_score


def calculate_accuracy_score(data_series: pd.Series, accuracy_options: AccuracyCalculationOptions, comparison_data: pd.DataFrame = pd.DataFrame(None)) -> float:

    window_length, approximation_curve, z_score = general_dispersion_analysis(
        data_series, accuracy_options)

    rate_missing_data, _ = detect_missing_data(data_series)
    rate_outliers_intrinsic, _, _ = detect_outliers_intrinsic(
        data_series, accuracy_options, approximation_curve, z_score)
    dispersion_score = get_dispersion_stats(
        data_series, approximation_curve, z_score, accuracy_options, window_length)

    accuracy_score = ((1 - rate_missing_data * accuracy_options.weights.missing_data) +
                      (1 - rate_outliers_intrinsic * accuracy_options.weights.outliers_intrinsic) +
                      (dispersion_score * accuracy_options.weights.dispersion_intrinsic)) / (accuracy_options.weights.sum_weights())

    return accuracy_score


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

        accuracy_options = AccuracyCalculationOptions()

        accuracy_score = calculate_accuracy_score(
            data_series, accuracy_options)

        print(f'Genauigkeits-Score: {accuracy_score:.2f}')

    # main()

    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"
    accuracy_options = AccuracyCalculationOptions(plot_outliers=True)
    comparison_data_options = ComparisonDataExtractionOptions()

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)

    score = determine_accuracy(data_series, accuracy_options, comparison_data)

    print(f'Genauigkeits-Score: {score:.2f}')
