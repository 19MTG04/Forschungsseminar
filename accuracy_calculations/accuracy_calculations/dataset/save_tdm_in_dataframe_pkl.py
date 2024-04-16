import numpy as np
import pandas as pd
from tqdm import tqdm
import tdm_loader

from accuracy_calculations.misc.path_helper import get_project_root


if __name__ == '__main__':
    filepath_loading = get_project_root() / 'dataset' / 'Prueflauf_23065.tdm'
    data_file = tdm_loader.OpenFile(filepath_loading)
    for observation_feature in data_file.channel_dict(21).keys():
        filepath_saving = get_project_root() / 'dataset' / \
            f'{observation_feature}.pkl'

        # Anzahl der gewünschten Zeilen im DataFrame
        number_rows_in_data = data_file.no_channel_groups()

        # Liste von DataFrames für jede Zeile
        dfs = []

        # Iteration über die Anzahl der Zeilen
        for i in tqdm(range(number_rows_in_data)):
            # Daten auslesen
            try:
                array_values = data_file.channel(i, observation_feature)
            except IndexError:
                continue

            # Erstellen eines temporären DataFrames für jedes Array
            temp_df = pd.DataFrame([array_values], index=[f"Zeitreihe {i}"],
                                   columns=range(len(array_values)))

            # Hinzufügen des temporären DataFrames zur Liste
            dfs.append(temp_df)

        # Verbinden aller DataFrames in der Liste zu einem einzigen DataFrame
        df_of_observated_feature = pd.concat(dfs, axis=0)
        df_of_observated_feature.to_pickle(filepath_saving, compression='gzip')

        print(df_of_observated_feature)
