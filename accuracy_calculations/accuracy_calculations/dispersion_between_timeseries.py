from typing import Tuple
import numpy as np
import pandas as pd


from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.statistical_analysis_helper import optimize_window_length, analyse_dispersion

# TODO: Umschreiben, dass es für den vergleich mehrerer Zeitreihen gilt.


def analyse_dispersion_intrinsic(data_series: pd.Series, options: AccuracyCalculationOptions) -> Tuple[float, pd.Series]:
    window_length = optimize_window_length(data_series)
    approximation_curve, z_score = analyse_dispersion(
        data_series, window_length=window_length, options=options)

    dispersion_score = get_dispersion_stats(
        z_score, options)

    return dispersion_score, approximation_curve


def get_dispersion_stats(z_score: pd.Series, options: AccuracyCalculationOptions) -> float:
    z_score_normalized = abs(z_score) / options.threshold_outliers
    z_score_clipped = z_score_normalized.clip(0, 1)

    dispersion_score = sum(np.cos(np.pi / 2 * z_score_clipped)
                           ) / len(z_score_clipped.dropna())

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
