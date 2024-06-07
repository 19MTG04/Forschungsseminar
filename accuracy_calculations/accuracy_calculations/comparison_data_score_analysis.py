import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions

from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions


def get_dispersion_and_outlier_score_for_comparison_data(data_series: pd.Series, comparison_data: pd.DataFrame, options: AccuracyCalculationOptions) -> Tuple[float, float, int]:
    data_points_per_timestep, weights_for_each_timestep, z_score = comparison_data_analysis(
        data_series, comparison_data, options)

    dispersion_score_comparison = get_dispersion_stats_for_comparison(
        weights_for_each_timestep, z_score, comparison_data, options)

    number_outliers_comparison, outlier_score_comparison = get_outlier_stats_for_comparison(
        data_series, options, data_points_per_timestep, z_score)

    return dispersion_score_comparison, outlier_score_comparison, number_outliers_comparison


def get_outlier_stats_for_comparison(data_series: pd.Series, options: AccuracyCalculationOptions, data_points_per_timestep: pd.Series, z_score: pd.Series) -> Tuple[int, float]:
    # Wenn mindestens 5 Vergleichsdaten vorliegen und der Wert dennoch weit abweicht, liegt ein Ausreißer vor.
    number_outliers_comparison = ((z_score > options.threshold_outliers) & (
        data_points_per_timestep >= options.minimum_number_of_comparison_data)).sum()

    # Der Score muss zwischen 0 und 1 liegen.
    outlier_score_comparison = np.clip(
        1 - options.multiplicator_outliers * number_outliers_comparison / len(data_series), 0, 1)

    return number_outliers_comparison, outlier_score_comparison


def get_dispersion_stats_for_comparison(weights_for_each_timestep: pd.Series, z_score: pd.Series, comparison_data: pd.DataFrame, options: AccuracyCalculationOptions) -> float:
    # Damit Ausreißer nicht bei Streuung und Ausreißer schlecht bewertet werden, werden sie hier ignoriert
    z_score_wo_outliers = z_score.mask(z_score > options.threshold_outliers, 0)

    # Die Ausnutzung des Toleranzbandes wird auf diese Weise bestimmt
    z_score_normalized = z_score_wo_outliers / options.threshold_outliers
    z_score_exploitation = (1 - (z_score_normalized * weights_for_each_timestep).sum() /
                            weights_for_each_timestep.sum())

    # Bezugsgröße um zu referenzieren, wie groß ide Größenordnung der Daten ist
    data_magnitude_reference = comparison_data.mean(axis=0).abs().max()

    # Straffaktor, wenn die Standardabweichungen in den Vergleichsdaten sehr groß sind
    penalty_magnitude_std = data_magnitude_reference / \
        (comparison_data.std(axis=0)**2 + data_magnitude_reference)

    dispersion_score_comparison = z_score_exploitation * penalty_magnitude_std

    return dispersion_score_comparison


def plot_dispersion_comparison(data_series: pd.Series, comparison_data_std: pd.Series, comparison_data_mean: pd.Series, options: AccuracyCalculationOptions) -> None:
    # Berechnung der oberen und unteren Grenzen des Toleranzbands
    upper_bound = comparison_data_mean + 2.58 * comparison_data_std
    lower_bound = comparison_data_mean - 2.58 * comparison_data_std

    # Plot der Datenreihe
    plt.plot(comparison_data_mean.index, data_series,
             label='Relevante Datenserie')

    if options.threshold_outliers == 2.58:
        label = 'Toleranzband 99% Konfidenzintervall'
    elif options.threshold_outliers == 1.96:
        label = 'Toleranzband 99% Konfidenzintervall'
    else:
        raise ValueError(
            'Threshold richtig angegeben? Breite des Toleranzbandes nicht explizit hinterlegt.')
    # Plot des oberen Toleranzbands
    plt.plot(comparison_data_mean.index, upper_bound,
             label=f'{label}', linestyle='--', color='red')
    plt.plot(comparison_data_mean.index, lower_bound,
             linestyle='--', color='red')

    plt.legend()
    plt.show()


def comparison_data_analysis(data_series: pd.Series, comparison_data: pd.DataFrame, options: AccuracyCalculationOptions) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Diese Funktion soll den z-Score für jeden Zeitpunkt der Vergleichsdaten bestimmen.
    Dieser Score gibt an, um wie viele Standardabweichungen der Wert der Zeitreihe vom Mittelwert der Vergleichsdaten abweicht.
    Außerdem werden die Anzahl der Vergleichsdaten pro Zeitpunkt und die daraus resultierende Gewichtung zurückgeführt.

    Wenn die entsprechende Options aktiviert ist, soll geplottet werden, wie die relevante Datenreihe im Vergleich zu den Vergleichsdaten aussieht.
    """
    comparison_data_std = comparison_data.std(axis=0).astype(float)
    comparison_data_mean = comparison_data.mean(axis=0).astype(float)
    data_points_per_timestep = comparison_data.notna().sum(axis=0)

    # Formel für beschränktes Wachstum.
    # Bei der Mindestanzahl an Vergleichsdaten wird der Score 0.25 vergeben, darüber hinaus nähert sich der Score 2 an.
    weights_for_each_timestep = 2 - \
        (2 - 0.25) * np.exp(-0.05 * (data_points_per_timestep -
                                     options.minimum_number_of_comparison_data))

    # Bei weniger Vergleichszeitreihen soll das Geicht 0 sein.
    weights_for_each_timestep = weights_for_each_timestep.mask(  # type: ignore
        weights_for_each_timestep < 0.25, 0)

    z_score = ((data_series.values - comparison_data_mean) /
               comparison_data_std).fillna(0).abs()  # type: ignore

    if options.plot_comparison_data:
        plot_dispersion_comparison(
            data_series, comparison_data_std, comparison_data_mean, options)

    return data_points_per_timestep, weights_for_each_timestep, z_score


if __name__ == '__main__':
    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"
    comparison_data_options = ComparisonDataExtractionOptions()
    accuracy_options = AccuracyCalculationOptions()

    relevant_data, comparison_dataframe = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)
    get_dispersion_and_outlier_score_for_comparison_data(
        relevant_data, comparison_dataframe, accuracy_options)
