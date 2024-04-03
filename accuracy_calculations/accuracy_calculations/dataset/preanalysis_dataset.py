import tdm_loader

from accuracy_calculations.misc.path_helper import get_project_root

filepath = get_project_root() / 'dataset' / 'Prueflauf_23065.tdm'
data_file = tdm_loader.OpenFile(filepath)
print(data_file)

# Bei 0, 1, 2, 3, 4, 6, 235 sind keine Daten
# In Spalte 29 fehlen bei 5, 7 Daten (Indizes beginnen bei 0!)
# In Spalte 30 und 31 fehlen bei 5, 7, 8, 9, 10, 11, 12, 13, 14 Daten (Indizes beginnen bei 0!)

# Spalte 32, 33, 34, 35 existieren nur bei 21, 26, 34 (Indizes beginnen bei 0!)

for i in range(0, 328):
    for n in range(31, 32):
        try:
            data_file.channel(i, n)
        except (IndexError):
            print(f'{i}, {n}')

# TODO: Wo ist zB das Drehmoment?
# TODO: Beliebiges Drehmoment extrahieren und visualisieren
# TODO: Länge der Drehmomenten Zeitreihen miteinander vergleichen. Wie lang sind die so? Welche sind vllt vergleichbar?
