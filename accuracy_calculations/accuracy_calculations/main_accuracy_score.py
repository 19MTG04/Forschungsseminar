import pandas as pd
import numpy as np
from accuracy_calculations.missing_data import detect_missing_data
from accuracy_calculations.outliers_no_comp import detect_outliers_intrinsic
from accuracy_calculations.dispersion_no_comp import analyse_dispersion_intrinsic
from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions

# TODO: Ich nutze nicht die Berechnung der Standardabweichung für eine Stichprobe bisher. Sollte ich? (ddof=0 im Moment, bei Testdaten besser)
# TODO: Im Bericht steht als Konfidenzintervall 95%. Eher 99% nutzen wie hier im Code oder?
# TODO: 2.1.5 ist im Bericht tendenziell falsch mit den Formeln. Pi/2 und besser beschreiben wie es im Code ist.
# TODO: NaN-Handling ist bisher nicht explizit betrachtet worden!
# TODO: Womöglich ist das Handling mit pd.tseries besser als mit pd.Series!


def calculate_accuracy_score(data_series: pd.Series, comparison_data: pd.DataFrame, accuracy_options: AccuracyCalculationOptions) -> float:
    rate_missing_data, _ = detect_missing_data(data_series)
    rate_outliers_intrinsic, _, _, _ = detect_outliers_intrinsic(
        data_series, accuracy_options)
    dispersion_score, _ = analyse_dispersion_intrinsic(
        data_series, accuracy_options)

    accuracy_score = ((1 - rate_missing_data * accuracy_options.weights.missing_data) +
                      (1 - rate_outliers_intrinsic * accuracy_options.weights.outliers_intrinsic) +
                      (dispersion_score * accuracy_options.weights.dispersion_intrinsic)) / (accuracy_options.weights.sum_weights())

    return accuracy_score


if __name__ == '__main__':
    def main():
        N = 1000
        np.random.seed(1)
        df = pd.DataFrame(
            {'test_data': np.sin(np.linspace(0, 10, num=N)) + np.random.normal(scale=0.6, size=N)})
        data_series = df['test_data']

        options = AccuracyCalculationOptions()

        accuracy_score = calculate_accuracy_score(
            data_series, pd.DataFrame(None), options)

        print(f'Genauigkeits-Score: {accuracy_score:.2f}')

    main()
