import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
import warnings

from accuracy_calculations.misc.path_helper import get_project_root


def sanity_check_origin_data(time_comp_unfiltered, step_unfiltered, rpm_unfiltered):
    if not len(time_comp_unfiltered) == len(step_unfiltered) == len(rpm_unfiltered):
        raise ValueError(
            f"Sind Sie sicher, dass die Ausgangsdaten zusammengehörig sind? Die Länge der Serien ist unterschiedlich ({len(time_comp_unfiltered)} vs. {len(step_unfiltered)}).")


def synchronize_indices(*dfs):
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    synchronized_dfs = [df.loc[common_index] for df in dfs]
    return synchronized_dfs


def remove_nan_entries_from_series(series1, series2, series3):
    # Kombinieren Sie beide Serien zu einem DataFrame, um den Index zu synchronisieren
    df = pd.concat([series1, series2, series3], axis=1)

    # Entfernen Sie Zeilen, die NaN-Werte enthalten
    df = df.dropna()

    # Teilen Sie das DataFrame in aktualisierte Serien auf
    updated_series1 = df.iloc[:, 0].reset_index(drop=True)
    updated_series2 = df.iloc[:, 1].reset_index(drop=True)
    updated_series3 = df.iloc[:, 2].reset_index(drop=True)

    return updated_series1, updated_series2, updated_series3


def create_comparison_data(time_comp: pd.Series, time_df: pd.DataFrame, n_df: pd.DataFrame, step: pd.Series, step_df: pd.DataFrame, mean_diff: float):
    filtered_df = pd.DataFrame(index=time_df.index, columns=time_comp)

    for time_index, time in time_comp.items():
        time_diff_mask = (time_df.sub(time, axis=1).abs() <= 0.1 * mean_diff)

        step_mask = (step_df == step[time_index])
        mask = time_diff_mask & step_mask

        filtered_values = n_df.where(mask)

        # Überprüfen, ob mindestens ein gültiger Wert vorhanden ist, bevor die Summe berechnet wird
        valid_values_mask = filtered_values.notna().any(axis=1)
        filtered_df.loc[valid_values_mask,
                        time] = filtered_values.sum(axis=1, skipna=True)

    return filtered_df


def extract_series_from_dfs(index_name, *dfs):
    extracted_values = []
    modified_dfs = []

    for df in dfs:
        if index_name in df.index:
            extracted_values.append(df.loc[index_name])
            modified_df = df.drop(index_name)
            modified_dfs.append(modified_df)
        else:
            raise ValueError(
                f"Die Zeitreihe {index_name} existiert nicht im Dataframe.")

    extracted_series = pd.Series(extracted_values, index=[
                                 df.index.name for df in dfs])
    return extracted_series, modified_dfs


def calculate_mean_time_difference(time_comp):
    mean_diff = time_comp.diff().mean()
    return mean_diff


def clean_nan_columns(df):
    columns_with_only_nan = df.columns[df.isnull().all()]
    df.drop(columns=columns_with_only_nan, inplace=True)


def slice_dataframe(df1, df2, df3, A_value, B_value) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if B_value <= A_value:
        raise ValueError(
            "Die obere zeitliche Grenze muss über der unteren liegen.")

    A_bool = (df1 >= A_value).any(axis=None)
    B_bool = (df1 < B_value).any(axis=None)
    if not A_bool or not B_bool:
        warnings.warn("Das hätte nicht passieren sollen.", UserWarning)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Index des letzten auftretens von A oder größer
    A_indices = (df1 >= A_value).idxmax(axis=1)
    A_indices_filtered = A_indices[(df1 >= A_value).any(axis=1)]

    # Index des ersten Auftretens von B in jeder Zeile finden
    B_indices = (df1 >= B_value).idxmax(axis=1)
    B_indices_filtered = B_indices[B_indices != 0]

    # Index des ersten Auftretens von B in jeder Zeile finden
    # Falls alle Werte in B_indices Nullen sind, setze den Index auf die maximale Spaltenzahl
    B_indices_max = B_indices_filtered.max() if len(
        B_indices_filtered) > 0 else len(df1.columns)

    # DataFrame slicen
    result_df1 = df1.loc[:, A_indices_filtered.min():B_indices_max]
    result_df2 = df2.loc[:, A_indices_filtered.min():B_indices_max]
    result_df3 = df3.loc[:, A_indices_filtered.min():B_indices_max]

    return result_df1, result_df2, result_df3


def delete_rows_by_names(df, *args):
    # Überprüfen, ob args nicht leer ist
    if not args:
        return df

    arg_list = []
    for i in args:
        arg_list.append(f'Zeitreihe {i}')

    # Löschen der Zeilen basierend auf den angegebenen Zeilennamen
    df_filtered = df.drop(index=(x for x in arg_list), errors='ignore')

    return df_filtered


if __name__ == '__main__':
    def main():
        time_comp = pd.Series(
            [245.5, 502.2, 747.9, 1000, 1250, 1500])
        step = pd.Series([1, 1, 1, 1, 2, 2])
        rpm = pd.Series([10, 20, 50, 100, 1000, 1000])
        sanity_check_origin_data(time_comp, step, rpm)
        time_comp, step, rpm = remove_nan_entries_from_series(
            time_comp, step, rpm)

        mean_diff = calculate_mean_time_difference(time_comp)

        time_df_init = pd.DataFrame([[250, 500, 750, 1000, 1250, 1500, 1800, np.nan],
                                    [125, 250, 375, 500, 750, 1000, 1200, np.nan],
                                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan],
                                    [500, 1000, 1500, 2000, np.nan, np.nan, np.nan, np.nan]], index=['Zeitreihe 1', 'Zeitreihe 2', 'Zeitreihe 3', 'Zeitreihe 4'])

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

        time_df_sliced, step_df_sliced, n_df_sliced = slice_dataframe(
            time_df, step_df, n_df, 500, 800)

        filtered_df = create_comparison_data(
            time_comp, time_df_sliced, n_df_sliced, step, step_df_sliced, mean_diff)
        print(filtered_df)

    # main()

    channel_group = 7

    # 3 Dataframes laden
    file_loading = get_project_root() / 'dataset'
    time_frame_initial = pd.read_pickle(
        file_loading / 'Zeit.pkl', compression='gzip')
    step_frame_initial = pd.read_pickle(
        file_loading / 'Prüfschritt.pkl', compression='gzip')
    rpm_frame_initial = pd.read_pickle(
        file_loading / 'Antrieb 1  Drehzahl.pkl', compression='gzip')

    # Extrahieren der Reihe, die betrachtet werden soll
    series_channel_group, modified_dfs = extract_series_from_dfs(
        f'Zeitreihe {channel_group}', time_frame_initial, step_frame_initial, rpm_frame_initial)
    time_channel_group, step_channel_group, rpm_channel_group = series_channel_group
    time_frame, step_frame, rpm_frame = modified_dfs
    clean_nan_columns(time_frame)

    # TODO: Diese Lösung muss besser sein. Automatisches generieren der Liste?
    # TODO: Soll eine der Listen untersucht werden, brauche ich auch dafür eine Lösung!
    time_frame = delete_rows_by_names(
        time_frame, 175, 202, 204, 206, 208, 210, 213, 215, 217)

    # Sanity check
    sanity_check_origin_data(
        time_channel_group, step_channel_group, rpm_channel_group)

    # Remove NaN
    time_channel_group, step_channel_group, rpm_channel_group = remove_nan_entries_from_series(
        time_channel_group, step_channel_group, rpm_channel_group)

    mean_diff = calculate_mean_time_difference(time_channel_group)

    # Indizes synchronisieren
    time_df_synchronized, step_df_synchronized, n_df_synchronized = synchronize_indices(
        time_frame, step_frame, rpm_frame)

    # Dataframes einmalig reduzieren, bezüglich der Zeitgrenzen
    time_df, step_df, n_df = slice_dataframe(
        time_df_synchronized, step_df_synchronized, n_df_synchronized, time_channel_group.iloc[0] - mean_diff * 0.1, time_channel_group.iloc[-1] + mean_diff * 0.1)

    df_list = []
    window_size = 750
    for i in tqdm(range(int(np.ceil(len(time_channel_group) / window_size)))):
        time_series = time_channel_group[window_size *
                                         i: window_size * (i + 1)]
        step_series = step_channel_group[window_size *
                                         i: window_size * (i + 1)]

        time_df, step_df, n_df = slice_dataframe(
            time_df_synchronized, step_df_synchronized, n_df_synchronized, time_series.iloc[0] - mean_diff * 0.1, time_series.iloc[-1] + mean_diff * 0.1)
        filtered_df = create_comparison_data(
            time_series, time_df, n_df, step_series, step_df, mean_diff)
        df_list.append(filtered_df)

    comparison = pd.concat(df_list, axis=1)
    print(comparison)
