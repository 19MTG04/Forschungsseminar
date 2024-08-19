import os
import re
import pandas as pd
from tqdm import tqdm

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.outliers_no_comp import detect_outliers_intrinsic
from accuracy_calculations.statistical_analysis_helper import general_dispersion_analysis, return_without_outliers
from base_library.misc.path_helper import get_project_root


def cleanse_data(observation_feature: str, df_of_observed_feature_partly_cleansed: pd.DataFrame) -> None:

    # Pfad zum Ordner mit den Dateien, die vorhanden sein müssen, um vollständig bereinigen zu können.
    results_path = get_project_root() / 'results'

    # Dateinamen, die erwartet werden, unter Einbeziehung des observation_feature
    expected_patterns = [
        rf'DS1_rawdata_Cbetr_{observation_feature}.*\.csv',
        rf'DS1_rawdata_Cwaelz_{observation_feature}.*\.csv',
        rf'DS2_rawdata_Cbetr_{observation_feature}.*\.csv',
        rf'DS2_rawdata_Cwaelz_{observation_feature}.*\.csv',
        rf'DS3_rawdata_Cbetr_{observation_feature}.*\.csv',
        rf'DS3_rawdata_Cwaelz_{observation_feature}.*\.csv'
    ]

    # Alle Dateien im Ordner auflisten
    files_in_folder = os.listdir(results_path)

    # Gefundene passende Dateien
    matching_files = []

    # Überprüfen, ob jede erwartete Datei vorhanden ist
    for pattern in expected_patterns:
        for csv_file in files_in_folder:
            if re.match(pattern, csv_file):
                matching_files.append(csv_file)
                break

    # Sicherstellen, dass der Ordner existiert, falls nicht, erstelle ihn
    filepath_saving = get_project_root() / "dataset_cleansed"
    os.makedirs(filepath_saving, exist_ok=True)

    # Wenn alle 6 Dateien vorhanden sind, kann vollständig bereinigt werden
    if len(matching_files) == len(expected_patterns):
        print("Alle erforderlichen Dateien sind vorhanden. Neben der intrinsischen Säuberung der Ausreißer aller Zeitreihen, können auch die dauerhaft stark abweichenden Zeitreihen eliminiert werden.")
        # Laden der 6 csv Dateien
        csv_data = {}
        for csv_file_matching in matching_files:
            df_name = f'{csv_file_matching.split('_')[0]}_{
                csv_file_matching.split('_')[2]}'
            csv_data[df_name] = pd.read_csv(
                os.path.join(results_path, csv_file_matching), index_col=0)

        # Alle Indexe finden, bei denen der Ausreißerscore im Vergleich kleiner 0,1 war.
        indexes_of_timeseries_to_exclude = set()
        for df in csv_data.values():
            indexes = df.columns[df.loc['Outliers comparison'] < 0.1]
            indexes_of_timeseries_to_exclude.update(indexes)

        # Bereinigen der Daten durch die gefundenen Indizes
        df_of_observed_feature = df_of_observed_feature_partly_cleansed[~df_of_observed_feature_partly_cleansed.index.isin(
            indexes_of_timeseries_to_exclude)]

        filename = f"{observation_feature}.pkl"
    else:
        # Sind nicht alle Dateien vorhanden, kann nur der Zwischenstand zwischengespeichert werden.
        # Ausreißer wurden bisher also ersetzt, vollständig ausreißende Reihen sind jedoch weiterhin vorhanden.
        print("Nicht alle erforderlichen Dateien sind vorhanden. Es wird nur ein bereinigter Zwischenstand gespeichert.")
        filename = f"{observation_feature}_partly_cleansed.pkl"
        df_of_observed_feature = df_of_observed_feature_partly_cleansed

    # Vervollständigen des Dateinamens und speichern
    file_path = os.path.join(filepath_saving, filename)
    df_of_observed_feature.to_pickle(file_path, compression='gzip')


def erase_intrinsic_outliers(observation_feature: str, accuracy_options: AccuracyCalculationOptions) -> pd.DataFrame:
    #  Dies sind die Zeitreihen, die in einem der drei Datensätze entweder im Feature Cwälz oder Cbetr vorkommen und länger als 10 Sekunden (also zumindest potenziell valide) sind.
    all_relevant_channel_groups = [11, 10, 9, 8, 14, 12, 13, 20, 18, 19, 16, 22,
                                   23, 25, 24, 105, 107, 104, 106, 98, 100, 101, 108, 111, 126, 112, 114, 102, 96, 121, 123, 128, 151, 148, 146, 143, 141, 139, 137, 133, 131, 140, 142, 144, 130, 147, 150, 149, 152, 154, 155, 156, 160, 157, 159, 158, 161, 170, 167, 202, 200, 255, 259, 260, 262, 263, 267, 265, 266, 264, 261, 268, 269, 270, 327, 326, 325, 324, 322, 27, 28, 32, 29, 35, 37, 40, 51, 50, 36, 45, 52, 44, 39, 42, 41, 46, 47, 38, 48, 49, 53, 94, 54, 93, 91, 90, 88, 86, 56, 57, 82, 78, 76, 73, 71, 60, 64, 62, 61, 63, 67, 70, 69, 72, 74, 59, 77, 79, 81, 83, 85, 58, 84, 87, 89, 55, 92,
                                   95, 124, 125, 120, 97, 99, 119, 117, 115, 113, 110, 127, 109, 168, 169, 162, 163, 166, 165, 171, 230, 228, 172, 223, 174, 221, 219, 217, 215, 213, 175, 210, 208, 206, 204, 198, 197, 195, 193, 191, 189, 187, 185, 183, 180, 181, 178, 177, 179, 176, 182, 184, 186, 188, 190, 192, 194, 196, 232, 199, 201, 203, 207, 209, 212, 211, 214, 216, 218, 220, 173, 224, 227, 226, 233, 234, 236, 246, 237, 238, 243, 241, 239, 240, 242, 247, 248, 252, 249, 250, 251, 253, 314, 313, 311, 310, 308, 309, 276, 272, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 290, 287, 285, 284, 282, 281, 280, 279, 277, 278, 274, 273, 275]

    # Anzahl der gewünschten Zeilen im DataFrame
    number_rows_in_data = len(all_relevant_channel_groups)

    # Liste von DataFrames für jede Zeile
    dfs = []
    file_loading = get_project_root() / 'dataset'
    observed_feature_frame_saved = pd.read_pickle(
        file_loading / f'{observation_feature}.pkl', compression='gzip')

    for channel_group in tqdm(sorted(all_relevant_channel_groups)):
        observed_data_channel_group = observed_feature_frame_saved.loc[
            f'Zeitreihe {channel_group}']
        data_series = observed_data_channel_group.dropna()

        window_length, approximation_curve, z_score = general_dispersion_analysis(
            data_series, accuracy_options)
        outlier_score, _, _, _ = detect_outliers_intrinsic(
            data_series, accuracy_options, approximation_curve, z_score)
        data_without_outlieres = return_without_outliers(
            data_series, approximation_curve, z_score, accuracy_options)

        # Erstellen eines temporären DataFrames für jedes Array
        temp_df = pd.DataFrame([data_without_outlieres], index=[f"Zeitreihe {channel_group}"],
                               columns=range(len(data_without_outlieres)))

        dfs.append(temp_df)

    # Zusammenfügen zu einem DataFrame, das alle bereinigten Zeitreihen als Zeilennamen enthält
    df_of_observed_feature = pd.concat(dfs, axis=0)
    print(df_of_observed_feature)
    return df_of_observed_feature


if __name__ == '__main__':
    observation_feature = "Antrieb 1  Drehzahl"
    accuracy_options = AccuracyCalculationOptions()

    df_of_observed_feature_partly_cleansed = erase_intrinsic_outliers(
        observation_feature, accuracy_options)

    # # Außer wenn man schonmal Zwischengespeichert hat, dann direkt:
    # df_of_observed_feature_partly_cleansed = pd.read_pickle(
    #     get_project_root() / 'dataset_cleansed' / f'{observation_feature}_partly_cleansed.pkl', compression='gzip')

    cleanse_data(observation_feature, df_of_observed_feature_partly_cleansed)
