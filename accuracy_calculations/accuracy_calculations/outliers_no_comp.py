from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zscore(data_series: pd.Series, window_length: int, threshold: float = 2.58) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ gemäß https://stackoverflow.com/questions/75938497/outlier-detection-of-time-series-data

    Erklärungen:
    Ausreißer, wenn der Wert mehr als 2.58 mal die Standardabweichung vom moving average abweicht (99% Konfidenzintervall).
    Für den moving average werden n=window_length Werte genutzt und die Mitte des Fensters betrachtet.
    Für den ersten Wert im average werden bspw. die ersten 25 Werte (wenn window_length = 50) der Serie genommen und durch 25 geteilt,
    da die weiteren 25 Werte, die genutzt werden würden, nicht existieren können (Quasi Indizes -25 bis -1) der Serie.

    Der z-Score ist der Wert der angibt, wie viele Standardabweichungen der Messwert vom moving average abweicht.
    Die within_threshold Variable ist eine Maske boolsche Maske, die angibt, ob der Wert innerhalb des Konfidenzintervalls liegt oder nicht.
    An den Stellen, an denen Sie False ergibt, liegt ein Ausreißer vor.
    """
    rolling_window = data_series.rolling(
        window=window_length, min_periods=1, center=True)
    moving_average = rolling_window.mean()
    moving_std = rolling_window.std(ddof=0)
    z_score = data_series.sub(moving_average).div(moving_std)
    within_threshold = z_score.between(-threshold, threshold)

    return z_score, moving_average, within_threshold


def get_outlier_stats(within_threshold: pd.Series) -> Tuple[int, float]:
    number_outlieres = len(within_threshold) - within_threshold.sum()
    outlier_rate = number_outlieres / len(within_threshold)
    return number_outlieres, outlier_rate


def return_without_outliers(data_series: pd.Series, moving_average: pd.Series, within_threshold: pd.Series) -> pd.Series:
    return data_series.where(within_threshold, moving_average)


if __name__ == '__main__':
    N = 1000
    np.random.seed(1)
    df = pd.DataFrame(
        {'MW': np.sin(np.linspace(0, 10, num=N))+np.random.normal(scale=0.6, size=N)})

    data_series = df['MW']

    z, avg, m = zscore(data_series, window_length=50)

    number_outliers, outlier_rate = get_outlier_stats(m)
    print(
        f'Anzahl der Ausreißer: {number_outliers}.')
    print(f'Anteil ausreißerfreier Daten: {(1 - outlier_rate) * 100}%')

    s = return_without_outliers(data_series, avg, m)

    ax = plt.subplot()

    df['MW'].plot(label='data')
    avg.plot(label='mean')
    df.loc[~m, 'MW'].plot(label='outliers', marker='o', ls='')
    avg[~m].plot(label='replacement', marker='o', ls='')
    plt.legend()
    plt.show()
