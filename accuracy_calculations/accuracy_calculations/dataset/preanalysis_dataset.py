import tdm_loader

from accuracy_calculations.misc.path_helper import get_project_root

filepath = get_project_root() / 'dataset' / 'Prueflauf_23065.tdm'
data_file = tdm_loader.OpenFile(filepath)
print(data_file)


channel_dict = {}

for i in range(0, 328):
    channel_count = data_file.no_channels(i)
    if channel_count not in channel_dict:
        channel_dict[channel_count] = []
    channel_dict[channel_count].append(i)

for key, value in channel_dict.items():
    print(f"Channels with {key} elements:", value)


# TODO: Wo ist zB das Drehmoment?
# TODO: Beliebiges Drehmoment extrahieren und visualisieren
# TODO: Länge der Drehmomenten Zeitreihen miteinander vergleichen. Wie lang sind die so? Welche sind vllt vergleichbar?
