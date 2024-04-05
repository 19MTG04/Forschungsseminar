from typing import Tuple, Any
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions


def regression_values(rolling_window: Any, options: AccuracyCalculationOptions) -> pd.Series:
    """
    "Moving Least Squares" gemäß https://luna.informatik.uni-mainz.de/compmod/cm1_assignments/02-Praktikum-Empirische-Modelle.md
    Berechnet die Werte in der Mitte der Regressionsfunktion für die Mitte des angegebenen gleitenden Fensters.
    """
    values = []
    for window_data in rolling_window:
        X = np.arange(len(window_data)).reshape(-1, 1)
        y = window_data.values.reshape(-1, 1)

        mode = options.mode_for_dispersion_identification

        # Grad des Regressionspolynoms gemäß der Optionen auslesen
        if mode == 'regression_deg_2':
            degree = 2
        elif mode == 'regression_deg_3':
            degree = 3
        else:
            raise ValueError(f"The given Dispersion Identification Mode {mode} \
                              is not implemented.")

        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Vorhersage für den mittleren Index des Fensters
        middle_index = len(window_data) // 2
        predicted_value = model.predict(
            poly_features.transform([[middle_index]]))[0][0]
        values.append(predicted_value)
    return pd.Series(values)


def optimize_window_length(data_series: pd.Series) -> int:

    def loss_function(window_length):
        rolling_window = data_series.rolling(
            window=round(float(window_length)), min_periods=1, center=True)
        moving_average = rolling_window.mean()

        whole_data_avg = data_series.mean()
        r2 = sum((moving_average - whole_data_avg)**2) / \
            sum((data_series - whole_data_avg)**2)

        difference_one_value_to_next_avg = moving_average.diff().abs().dropna().sum()
        difference_one_value_to_next_data = data_series.diff().abs().dropna().sum()
        change_rate = difference_one_value_to_next_avg / difference_one_value_to_next_data

        # Empirisch. R2 soll etwas stärker gewichtet sein, als die Glättung
        return -(change_rate - 1.3 * r2)**2

    # Minimierung der Kostenfunktion. R^2 soll möglichst groß sein, während eine Glatte Approximation vorliegen soll.
    result = minimize_scalar(loss_function, bounds=(
        5, len(data_series)), method='bounded', options={'maxiter': 100})
    optimal_window_length = round(result.x)

    return optimal_window_length


def analyse_dispersion(data_series: pd.Series, window_length: int, options: AccuracyCalculationOptions) -> Tuple[pd.Series, pd.Series]:
    """ gemäß https://stackoverflow.com/questions/75938497/outlier-detection-of-time-series-data

    Erklärungen:
    Ausreißer, wenn der Wert mehr als 2.58 mal die Standardabweichung von der Approximationskurve (Moving Average oder Moving Least Squared) abweicht (99% Konfidenzintervall).
    Für die Approximationskurve werden n=window_length Werte genutzt und die Mitte des Fensters betrachtet.
    Für den ersten Wert in der Approximation werden bspw. die ersten 25 Werte (wenn window_length = 50) der Serie genommen und durch 25 geteilt,
    da die weiteren 25 Werte, die genutzt werden würden, nicht existieren können (Quasi Indizes -25 bis -1) der Serie.

    Der z-Score ist der Wert der angibt, wie viele Standardabweichungen der Messwert von der Approximationskurve abweicht.
    Die within_threshold Variable ist eine Maske boolsche Maske, die angibt, ob der Wert innerhalb des Konfidenzintervalls liegt oder nicht.
    An den Stellen, an denen Sie False ergibt, liegt ein Ausreißer vor.
    """
    rolling_window = data_series.rolling(
        window=window_length, min_periods=1, center=True)
    if options.mode_for_dispersion_identification == 'mean':
        approximation_curve = rolling_window.mean()
    else:
        approximation_curve = regression_values(rolling_window, options)

    moving_std = rolling_window.std(ddof=0)

    # NaN können nur entstehen wenn durch 0 geteilt wird. Dann wird der z-Score auch mit 0 bewertet.
    z_score = data_series.sub(approximation_curve).div(moving_std).fillna(0)

    return approximation_curve, z_score


def get_outlier_mask(z_score: pd.Series, options: AccuracyCalculationOptions) -> pd.Series:
    within_threshold = z_score.between(-options.threshold_outliers,
                                       options.threshold_outliers)

    return within_threshold


def return_without_outliers(data_series: pd.Series, approximation_curve: pd.Series, z_score: pd.Series, options: AccuracyCalculationOptions) -> pd.Series:
    within_threshold = get_outlier_mask(z_score, options)
    return data_series.where(within_threshold, approximation_curve)
