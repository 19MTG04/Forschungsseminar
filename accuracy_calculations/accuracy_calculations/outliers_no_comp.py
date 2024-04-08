from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.statistical_analysis_helper import optimize_window_length, analyse_dispersion, get_outlier_mask


def detect_outliers_intrinsic(data_series: pd.Series, options: AccuracyCalculationOptions) -> Tuple[float, int, pd.Series, pd.Series]:
    window_length = optimize_window_length(data_series, options)
    approximation_curve, z_score = analyse_dispersion(
        data_series, window_length=window_length, options=options)

    number_outliers, outlier_rate, outlier_mask = get_outlier_stats(
        z_score, options)

    if options.plot_outliers:
        plot_outliers(data_series, approximation_curve, outlier_mask)

    return outlier_rate, number_outliers, approximation_curve, outlier_mask


def get_outlier_stats(z_score: pd.Series, options: AccuracyCalculationOptions) -> Tuple[int, float, pd.Series]:
    """ Bestimmung der Ausreißeranzahl und -rate.
    """
    within_threshold = get_outlier_mask(z_score, options)

    number_outlieres = len(within_threshold) - within_threshold.sum()
    outlier_rate = number_outlieres / len(within_threshold)
    return number_outlieres, outlier_rate, within_threshold


def plot_outliers(data_series: pd.Series, approximation_curve: pd.Series, outlier_mask: pd.Series) -> None:
    data_series.plot(label='data')
    approximation_curve.plot(label='approximation')
    data_series[~outlier_mask].plot(label='outliers', marker='o', ls='')
    approximation_curve[~outlier_mask].plot(
        label='possible replacement', marker='o', ls='')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    def main():
        N = 1000
        np.random.seed(1)
        df = pd.DataFrame(
            {'test_data': np.sin(np.linspace(0, 10, num=N)) + np.random.normal(scale=0.6, size=N)})
        data_series = df['test_data']

        options = AccuracyCalculationOptions()

        outlier_rate, num_outliers, approximation, outlier_mask = detect_outliers_intrinsic(
            data_series, options)

        print(
            f'Anzahl der Ausreißer: {num_outliers}')
        print(f'Anteil ausreißerfreier Daten: {(1 - outlier_rate) * 100}%')

        plot_outliers(data_series, approximation, outlier_mask)

    main()
