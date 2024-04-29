import numpy as np

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions

from base_library.extract_data_and_comparison_data import extract_data_and_comparison_data
from base_library.data_extraction_options import ComparisonDataExtractionOptions

# TODO Diese Funktion aufgliedern und schön machen!
# TODO: Spezifischere Optionen einführen.


def calculate_dispersion_of_comparion_data_score(data_series, comparison_data):
    comparison_data_std = comparison_data.std(axis=0).astype(float)
    comparison_data_mean = comparison_data.mean(axis=0).astype(float)
    data_points_per_timestep = comparison_data.notna().sum(axis=0)
    weights_for_each_timestep = 2 - \
        (2 - 0.25) * np.exp(-0.05 * (data_points_per_timestep-4))
    weights_for_each_timestep = weights_for_each_timestep.mask(
        weights_for_each_timestep < 0.25, 0)

    z_score = ((data_series.values - comparison_data_mean) /
               comparison_data_std).fillna(0).abs()
    z_score_clipped = np.clip(z_score, 0, 2.58)

    dispersion_score_comparison = (1 - (z_score_clipped * weights_for_each_timestep).sum() /
                                   weights_for_each_timestep.sum())
    number_outliers_comparison = ((z_score > 2.58) & (
        data_points_per_timestep >= 4)).sum()
    outlier_score_comparison = 1 - \
        100 * number_outliers_comparison / len(data_series)
    return dispersion_score_comparison, outlier_score_comparison, number_outliers_comparison


if __name__ == '__main__':
    channel_group = 214
    observation_feature = "Antrieb 1  Drehzahl"
    comparison_data_options = ComparisonDataExtractionOptions()

    data_series, comparison_data = extract_data_and_comparison_data(
        channel_group, observation_feature, comparison_data_options)
    calculate_dispersion_of_comparion_data_score(data_series, comparison_data)
