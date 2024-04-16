import pandas as pd
import numpy as np


def sanity_check_origin_data(time_comp_unfiltered, step_unfiltered):
    if not len(time_comp_unfiltered) == len(step_unfiltered):
        raise ValueError(
            f"Sind Sie sicher, dass die Ausgangsdaten zusammengehörig sind? Die Länge der Serien ist unterschiedlich ({len(time_comp_unfiltered)} vs. {len(step_unfiltered)}).")


def synchronize_indices(*dfs):
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    synchronized_dfs = [df.loc[common_index] for df in dfs]
    return synchronized_dfs


def remove_nan_entries_from_series(series1, series2):
    # Kombinieren Sie beide Serien zu einem DataFrame, um den Index zu synchronisieren
    df = pd.concat([series1, series2], axis=1)

    # Entfernen Sie Zeilen, die NaN-Werte enthalten
    df = df.dropna()

    # Teilen Sie das DataFrame in aktualisierte Serien auf
    updated_series1 = df.iloc[:, 0].reset_index(drop=True)
    updated_series2 = df.iloc[:, 1].reset_index(drop=True)

    return updated_series1, updated_series2


def create_filtered_df(time_comp: pd.Series, time_df: pd.DataFrame, n_df: pd.DataFrame, step: pd.Series, step_df: pd.DataFrame, mean_diff: float):
    filtered_df = pd.DataFrame(index=time_df.index, columns=time_comp)

    for index, time in enumerate(time_comp):
        time_diff_mask = (time_df.sub(time, axis=1).abs() <= 0.1 * mean_diff)
        step_mask = (step_df == step[index])
        mask = time_diff_mask & step_mask

        filtered_values = n_df.where(mask)

        # Überprüfen, ob mindestens ein gültiger Wert vorhanden ist, bevor die Summe berechnet wird
        valid_values_mask = filtered_values.notna().any(axis=1)
        filtered_df.loc[valid_values_mask,
                        time] = filtered_values.sum(axis=1, skipna=True)

    return filtered_df


if __name__ == '__main__':
    def main():
        time_comp_unfiltered = pd.Series(
            [245.5, 502.2, 747.9, 1003, 1251, 1500])
        step_unfiltered = pd.Series([1, 1, 1, 1, 2, 2])
        sanity_check_origin_data(time_comp_unfiltered, step_unfiltered)
        time_comp, step = remove_nan_entries_from_series(
            time_comp_unfiltered, step_unfiltered)

        mean_diff = time_comp.diff().mean()

        time_df_init = pd.DataFrame([[250, 500, 750, 1000, 1250, 1500],
                                    [125, 250, 375, 500, 750, 1000],
                                    [100, 300, 500, 700, 900, 1100],
                                    [500, 1000, 1500, 2000, np.nan, np.nan]], index=['Zeitreihe 1', 'Zeitreihe 2', 'Zeitreihe 3', 'Zeitreihe 4'])

        step_df_init = pd.DataFrame([[1, 1, 1, 2, 2, 2],
                                    [1, 1, 1, 1, 1, 1],
                                    [2, 2, 2, 2, 2, 2],
                                    [1, 1, 1, 1, 2, 2]], index=['Zeitreihe 1', 'Zeitreihe 2', 'Zeitreihe 3', 'Zeitreihe 4'])

        n_df_init = pd.DataFrame([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                                  [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
                                  [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
                                  [4.1, 4.2, 4.3, 4.4, 4.5, 4.6]], index=['Zeitreihe 1', 'Zeitreihe 2', 'Zeitreihe 3', 'Zeitreihe 4'])

        time_df, step_df, n_df = synchronize_indices(
            time_df_init, step_df_init, n_df_init)

        filtered_df = create_filtered_df(
            time_comp, time_df, n_df, step, step_df, mean_diff)
        print(filtered_df)
    main()

    # TODO: 3 Dataframes laden
    # TODO: Extrahieren der Reihe, die betrachtet werden soll
    # TODO: Dabei müssen die Daten auch aus den Dataframes entfernt werden
    # TODO: Sanity check
    # TODO: Remove NaN
    # TODO: Indizes synchronisieren
    # TODO: Vergleichsdaten erstellen
    # TODO: Vergleichsdaten analysieren. Habe ich alles richtig gemacht?
