from typing import Tuple, Any
import numpy as np
import pandas as pd


from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.statistical_analysis_helper import optimize_window_length, analyse_dispersion, return_without_outliers


def analyse_dispersion_intrinsic(data_series: pd.Series, options: AccuracyCalculationOptions) -> Tuple[float, pd.Series]:
    window_length = optimize_window_length(data_series)
    approximation_curve, z_score = analyse_dispersion(
        data_series, window_length=window_length, options=options)

    dispersion_score = get_dispersion_stats(
        data_series, approximation_curve, z_score, options, window_length)

    return dispersion_score, approximation_curve


def get_dispersion_stats(data_series: pd.Series, approximation: pd.Series,
                         z_score: pd.Series, options: AccuracyCalculationOptions,
                         window_length: int) -> float:
    data_without_outlieres = return_without_outliers(
        data_series, approximation, z_score, options)

    data_rolling_window = data_without_outlieres.rolling(
        window=window_length, min_periods=1, center=True)
    approximation_rolling_window = approximation.rolling(
        window=window_length, min_periods=1, center=True)

    difference_start_end_approx_windows = abs(approximation_rolling_window.apply(
        calculate_difference_window_start_end))
    absolute_difference_sum_data = data_rolling_window.apply(
        calculate_sum_absolute_difference_between_each_window_entry)

    # Dieses Verhältnis wird klein (nahe 0) für verrauschte und groß (nahe 1) für weniger verrauschte Zeitreihen.
    # Das liegt daran, dass die Differenz von Start und Ende der Fenster der Approximation relativ resistent gegen das Rauschen ist
    # und ein Maßstab dafür, wie viel Veränderung der Größenordnungen in den Daten rauschfrei zu erwarten ist.
    # Im Gegensatz dazu ist bei Betrachtung der Unterschiede zwischen jedem Wert in jedem Fenster der Realdaten
    # ein besonders großer Wert bei viel Rauschen zu erwarten.
    ratio_dispersion = difference_start_end_approx_windows.sum() / \
        absolute_difference_sum_data.sum()

    # Empirische Gleichung zur Bewertung des vorher eingeführten Verhältnisses.
    dispersion_score = (-np.exp(-4 * ratio_dispersion) + 1) / (-np.exp(-4) + 1)

    return dispersion_score


def calculate_difference_window_start_end(window: Any):
    difference = window.iloc[-1] - window.iloc[0]
    if np.isnan(difference):
        raise ValueError("Unhandeld NaN")
    return difference


def calculate_sum_absolute_difference_between_each_window_entry(window: Any):
    sum_abs_diff = window.diff().abs().sum()
    if np.isnan(sum_abs_diff):
        raise ValueError("Unhandeld NaN")
    return sum_abs_diff


if __name__ == '__main__':
    def main():
        N = 1000
        np.random.seed(1)
        df = pd.DataFrame(
            {'test_data': np.sin(np.linspace(0, 10, num=N)) + np.random.normal(scale=0.6, size=N)})
        data_series = df['test_data']

        options = AccuracyCalculationOptions()

        dispersion_score, _ = analyse_dispersion_intrinsic(
            data_series, options)

        print(f'Streuungs-Score: {dispersion_score:.2f}')

    main()
