import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
import warnings

from base_library.data_extraction_options import ComparisonDataExtractionOptions
from base_library.misc.path_helper import get_project_root


def sanity_check_origin_data(time_comp_unfiltered: pd.Series, step_unfiltered: pd.Series, observed_feature_unfiltered: pd.Series) -> None:
    if not len(time_comp_unfiltered) == len(step_unfiltered) == len(observed_feature_unfiltered):
        raise ValueError(
            f"Sind Sie sicher, dass die Ausgangsdaten zusammengehörig sind? Die Länge der Serien ist unterschiedlich ({len(time_comp_unfiltered)} vs. {len(step_unfiltered)}).")


def synchronize_indices(*dfs: pd.DataFrame) -> List[pd.DataFrame]:
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    synchronized_dfs = [df.loc[common_index] for df in dfs]
    return synchronized_dfs


def remove_nan_entries_from_series(series1: pd.Series, series2: pd.Series, series3: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Kombinieren Sie beide Serien zu einem DataFrame, um den Index zu synchronisieren
    df = pd.concat([series1, series2, series3], axis=1)

    # Entfernen Sie Zeilen, die NaN-Werte enthalten
    df = df.dropna()

    # Teilen Sie das DataFrame in aktualisierte Serien auf
    updated_series1 = df.iloc[:, 0].reset_index(drop=True)
    updated_series2 = df.iloc[:, 1].reset_index(drop=True)
    updated_series3 = df.iloc[:, 2].reset_index(drop=True)

    return updated_series1, updated_series2, updated_series3


def create_comparison_data_for_one_window(time_comp: pd.Series, time_df: pd.DataFrame, n_df: pd.DataFrame, step: pd.Series, step_df: pd.DataFrame, mean_diff: float) -> pd.DataFrame:
    filtered_df = pd.DataFrame(index=time_df.index, columns=time_comp)

    for time_index, time in time_comp.items():
        time_diff_mask = (time_df.sub(time, axis=1).abs() <= 0.1 * mean_diff)

        step_mask = (step_df == step[int(str(time_index))])
        mask = time_diff_mask & step_mask

        filtered_values = n_df.where(mask)

        # Überprüfen, ob mindestens ein gültiger Wert vorhanden ist, bevor die Summe berechnet wird
        valid_values_mask = filtered_values.notna().any(axis=1)
        filtered_df.loc[valid_values_mask,
                        time] = filtered_values.sum(axis=1, skipna=True)

    return filtered_df


def extract_series_from_dfs(index_name: str, *dfs: pd.DataFrame) -> Tuple[pd.Series, List[pd.DataFrame]]:
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


def calculate_mean_time_difference(time_comp: pd.Series) -> float:
    mean_diff = time_comp.diff().mean()
    return mean_diff


def clean_nan_columns(df: pd.DataFrame) -> None:
    columns_with_only_nan = df.columns[df.isnull().all()]
    df.drop(columns=columns_with_only_nan, inplace=True)


def slice_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, time_series: pd.Series, mean_difference: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    lower_bound = time_series.iloc[0] - mean_difference * 0.1
    upper_bound = time_series.iloc[-1] + mean_difference * 0.1
    if upper_bound <= lower_bound:
        raise ValueError(
            "Die obere zeitliche Grenze muss über der unteren liegen.")

    lower_bound_test = (df1 >= lower_bound).any(axis=None)
    upper_bound_test = (df1 < upper_bound).any(axis=None)
    if not lower_bound_test or not upper_bound_test:
        warnings.warn(
            "Hier gibt es offenbar keine Vergleichsdaten.", UserWarning)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, len(time_series)

    # Index des letzten auftretens von der unteren Grenze oder größer
    lower_bound_indices = (df1 >= lower_bound).idxmax(axis=1)
    # NaNs filtern
    lb_indiced_filtered = lower_bound_indices[(df1 >= lower_bound).any(axis=1)]
    lb_index_min = lb_indiced_filtered.min()

    # Index des ersten Auftretens von der oberen Grenze in jeder Zeile finden
    upper_bound_indices = (df1 >= upper_bound).idxmax(axis=1)
    ub_indiced_filtered = upper_bound_indices[upper_bound_indices != 0]

    # Index des ersten Auftretens von der oberen Grenze in jeder Zeile finden
    # Falls alle Werte in Nullen sind, wird die obere Grenze nicht erreicht und die maximale Spaltenzahl muss beachtet werden
    ub_index_max = ub_indiced_filtered.max() if len(
        ub_indiced_filtered) > 0 else len(df1.columns)

    # DataFrame slicen, sodass nur die zeitlichen interessanten WErte weiterhin betrachtet werden
    result_df1 = df1.loc[:, lb_index_min:ub_index_max]
    result_df2 = df2.loc[:, lb_index_min:ub_index_max]
    result_df3 = df3.loc[:, lb_index_min:ub_index_max]

    return result_df1, result_df2, result_df3, lb_index_min, ub_index_max


def delete_rows_with_different_frequency(df: pd.DataFrame, mean_diff: float) -> pd.DataFrame:
    diff_df = df.diff(axis=1).mean(axis=1)

    # Alle Indizes finden von Daten, bei denen die Messwertaufnahme 10% von der ursprünglichen, der halben oder der doppelten Frequenz abweicht.
    indices = diff_df[(abs(diff_df - mean_diff) > 0.1 * mean_diff) & (abs(diff_df - 2 * mean_diff) > 0.1 *
                                                                      2 * mean_diff) & (abs(diff_df - 0.5 * mean_diff) > 0.1 * 0.5 * mean_diff) | (diff_df.isna())].index
    df_filtered = df.drop(index=(x for x in indices))
    return df_filtered


def create_comparison_data_for_all_windows(time_channel_group: pd.Series, step_channel_group: pd.Series, mean_diff: float, time_df: pd.DataFrame, step_df: pd.DataFrame, oberserved_feature_df: pd.DataFrame, lb_index: int, ub_index: int, options: ComparisonDataExtractionOptions):

    df_list = []
    window_size = options.window_size

    print("Extraktion der Vergleichsdaten beginnt:")
    for i in tqdm(range(int(np.ceil(len(time_channel_group) / window_size)))):
        time_series = time_channel_group[window_size *
                                         i: window_size * (i + 1)]
        step_series = step_channel_group[window_size *
                                         i: window_size * (i + 1)]

        # Für jedes Fenster werden die Dataframes zur schnelleren Verarbeitung weiter unterteilt
        # Um bei sehr sehr großen Dataframes unnötige Indexsuche zu vermeiden,
        # wird auf diese Weise ein großzügiger Bereich vorgegeben, indem die entsprechend gesuchten Werte liegen werden.
        ub_logical = min(ub_index + 6 * (ub_index - lb_index),
                         time_df.shape[1])
        time_df_relevant, step_df_relevant, observed_feature_df_relevant, lb_index, ub_index = slice_dataframe(
            time_df.loc[:, lb_index:ub_logical], step_df.loc[:, lb_index:ub_logical], oberserved_feature_df.loc[:, lb_index:ub_logical], time_series, mean_diff)

        filtered_df = create_comparison_data_for_one_window(
            time_series, time_df_relevant, observed_feature_df_relevant, step_series, step_df_relevant, mean_diff)
        df_list.append(filtered_df)

    # Zusammenfügen der Vergleichsdaten aller Fenster
    comparison_data = pd.concat(df_list, axis=1)
    return comparison_data


def extract_data_and_comparison_data(channel_group: int, observation_feature: str, options: ComparisonDataExtractionOptions, ) -> Tuple[pd.Series, pd.DataFrame]:
    # Dataframes laden
    time_frame_saved, step_frame_saved, observed_feature_frame_saved = load_dataframe_data(
        observation_feature)

    # Extrahieren der relevanten Zeitreihe und löschen aus dem ursprünglichen Dataframe, sodass dort nur die Vergleichsdaten betrachtet werden können
    relevant_series_channel_group, modified_dfs = extract_series_from_dfs(
        f'Zeitreihe {channel_group}', time_frame_saved, step_frame_saved, observed_feature_frame_saved)
    time_channel_group, step_channel_group, observed_data_channel_group = relevant_series_channel_group
    time_frame, step_frame, observed_feature_frame = modified_dfs

    # Löschen von Spalten im Zeit-Dataframe ohne Daten zur nachträglich schnelleren Berechnung
    clean_nan_columns(time_frame)

    # Daten der extrahierten Serien müssen gleich lang sein!
    sanity_check_origin_data(
        time_channel_group, step_channel_group, observed_data_channel_group)

    # Löschen von NaNs aus den Serien. Wenn einer der Werte nicht vorliegt, können an dieser Stelle keine Vergleichsdaten berechnet werden
    time_channel_group, step_channel_group, observed_data_channel_group = remove_nan_entries_from_series(
        time_channel_group, step_channel_group, observed_data_channel_group)

    # Berechnen der durchschnittlichen Zeit zwischen zwei Messwertaufnahmen
    # Auf dieser Basis werden die Vergleichsdaten ausgewählt.
    mean_diff = calculate_mean_time_difference(time_channel_group)

    # Wenn die Frequenz der Datenerfassung in den Serien nicht gleich, doppelt oder halb so schnell ist, können diese Daten außen vor gelassen werden.
    # Dies spart gegebenenfalls viel Zeit bei der Vergleichsdatenfindung.
    if options.use_data_with_similar_frequency_only:
        time_frame = delete_rows_with_different_frequency(
            time_frame, mean_diff)

    # Löschen von Zeilen in den Vergleichsdaten, bei denen nicht alle Informationen zur Verfügung stehen.
    time_df_synchronized, step_df_synchronized, observed_feature_df_synchronized = synchronize_indices(
        time_frame, step_frame, observed_feature_frame)

    # Dataframes einmalig reduzieren, bezüglich der Zeitgrenzen
    time_df, step_df, observed_feature_df, lb_index, ub_index = slice_dataframe(
        time_df_synchronized, step_df_synchronized, observed_feature_df_synchronized, time_channel_group, mean_diff)

    comparison_data = create_comparison_data_for_all_windows(
        time_channel_group, step_channel_group, mean_diff, time_df, step_df, observed_feature_df, lb_index, ub_index, options)

    return observed_data_channel_group, comparison_data


def load_dataframe_data(observation_feature: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    file_loading = get_project_root() / 'dataset'
    time_frame_saved = pd.read_pickle(
        file_loading / 'Zeit.pkl', compression='gzip')
    step_frame_saved = pd.read_pickle(
        file_loading / 'Prüfschritt.pkl', compression='gzip')
    observed_feature_frame_saved = pd.read_pickle(
        file_loading / f'{observation_feature}.pkl', compression='gzip')

    return time_frame_saved, step_frame_saved, observed_feature_frame_saved


def extract_relevant_data(channel_group: int, observation_feature: str) -> pd.Series:
    file_loading = get_project_root() / 'dataset'
    observed_feature_frame_saved = pd.read_pickle(
        file_loading / f'{observation_feature}.pkl', compression='gzip')
    observed_data_channel_group = observed_feature_frame_saved.loc[
        f'Zeitreihe {channel_group}']
    observed_data = observed_data_channel_group.dropna()
    return observed_data


if __name__ == '__main__':

    channel_group_data = 200
    feature = "Antrieb 1  Drehzahl"
    comparison_options = ComparisonDataExtractionOptions()

    # Extrahieren der Reihe, die betrachtet werden soll
    channel_group_data_observed_feature, comparison = extract_data_and_comparison_data(
        channel_group_data, feature, comparison_options)
    print(comparison)
