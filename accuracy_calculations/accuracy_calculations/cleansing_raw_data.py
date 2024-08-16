import os
import pandas as pd
from tqdm import tqdm

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions
from accuracy_calculations.outliers_no_comp import detect_outliers_intrinsic
from accuracy_calculations.statistical_analysis_helper import general_dispersion_analysis, return_without_outliers
from base_library.misc.path_helper import get_project_root


if __name__ == '__main__':
    observation_feature = "Antrieb 1  Drehzahl"
    accuracy_options = AccuracyCalculationOptions()

    filename = f"{observation_feature}.pkl"
    filepath_saving = get_project_root() / "dataset_cleansed"
    file_path = os.path.join(filepath_saving, filename)

    # Sicherstellen, dass der Ordner existiert, falls nicht, erstelle ihn
    os.makedirs(filepath_saving, exist_ok=True)

    # Dies sind die Zeitreihen, die in einem der drei Datensätze entweder im Feature Cwälz oder Cbetr vorkommen und länger als 10 Sekunden (also zumindest potenziell valide) sind.
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

    df_of_observated_feature = pd.concat(dfs, axis=0)
    df_of_observated_feature.to_pickle(file_path, compression='gzip')

    print(df_of_observated_feature)
