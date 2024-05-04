from datetime import datetime
import tdm_loader

from base_library.misc.path_helper import get_project_root


if __name__ == '__main__':
    filepath_loading = get_project_root() / 'dataset' / 'Prueflauf_23065.tdm'
    data_file = tdm_loader.OpenFile(filepath_loading)
    liste = []
    for i in range(data_file.no_channel_groups()):
        liste.append([data_file.channel_group_name(i), i])
    sortiert = sorted(liste)

    daten = [datetime(2023, 9, 1, 1, 0, 0), datetime(2023, 9, 18, 14, 9, 0), datetime(2023, 9, 19, 10, 23, 0), datetime(2023, 9, 19, 10, 41, 0), datetime(2023, 9, 20, 8, 50, 0), datetime(2023, 9, 23, 18, 53, 0), datetime(2023, 9, 25, 16, 0, 0), datetime(2023, 9, 25, 16, 0, 59), datetime(2023, 9, 28, 16, 23, 0), datetime(2023, 9, 29, 16, 0, 0), datetime(2023, 10, 1, 0, 0, 0), datetime(2023, 10, 5, 8, 0, 0), datetime(
        2023, 10, 5, 14, 40, 0), datetime(2023, 10, 6, 7, 0, 0), datetime(2023, 10, 6, 22, 2, 0), datetime(2023, 10, 7, 2, 49, 0), datetime(2023, 10, 8, 15, 10, 0), datetime(2023, 10, 9, 22, 39, 0), datetime(2023, 10, 10, 7, 30, 0), datetime(2023, 10, 12, 14, 45, 0), datetime(2023, 10, 15, 16, 39, 0), datetime(2023, 10, 16, 6, 18, 0), datetime(2023, 10, 17, 7, 22, 0), datetime(2023, 12, 31, 23, 59, 0)]

    for i in range(len(daten) - 1):
        reference_date_davor = daten[i]
        reference_date_danach = daten[i+1]
        count = sum(1 for entry in sortiert if (datetime.strptime(
            entry[0], '%Y_%m_%d_%H_%M_%S') < reference_date_danach) & (datetime.strptime(
                entry[0], '%Y_%m_%d_%H_%M_%S') > reference_date_davor))
        print("Anzahl der Einträge nach dem", reference_date_davor, ":", count)
