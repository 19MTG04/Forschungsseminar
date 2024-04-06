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

    data_std = data_rolling_window.std()
    approximation_std = approximation_rolling_window.std()
    std_difference = (data_std - approximation_std).abs()
    mean_percentage_deviation = std_difference.mean() / approximation.max() * 100

    # 1 / (x + 1) als Formel zur Bewertung des Scores, sodass:
    # 0,01 % Abweichung zu ca. 0,99 als Score führen
    # 0,1 % Abweichung zu ca. 0,9 als Score führen
    # 1 % Abweichung zu ca. 0,5 als Score führen
    # 10 % Abweichung zu ca. 0,1 als Score führen
    dispersion_score = 1 / (mean_percentage_deviation + 1)

    return dispersion_score


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
