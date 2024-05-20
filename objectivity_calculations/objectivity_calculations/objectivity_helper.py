import pandas as pd
from typing import Tuple, List, Any
from copy import deepcopy

from base_library.data_extraction_options import ComparisonDataExtractionOptions


def extract_dataframes_for_objectivity_calculations(data_series: pd.Series, comparison_data: pd.DataFrame, comparison_data_options: ComparisonDataExtractionOptions) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Hier werden 2 DataFrames erstellt, gemäß der ComparisonDataExtractionOptions und den darin vorgegebenen Zeitfenstern.
    Für die Objektivität müssen jeweils zwei Dataframes zur Verfügung stehen, sodass man die Objektivität gegeneinander testen kann.
    Eines der DataFrames (das mit dem same_iterator) enthält unter anderem letztlich die Vergleichszeitreihe.
    Wenn ein anderer Datensatz angegeben ist (in den Optionen unter period_limitations_additional_dataset), 
        enthält der same_iterator alle Vergleichsdaten, aus der Zeitspanne period_limitations_same_dataset.
    Ist kein anderer Datensatz vorgegeben, enthält der same_iterator letztlich nur die Vergleichszeitreihe und die Vergleichsdaten sind im zweiten Iterator gespeichert.

    Args:
        data_series (pd.Series): Die Daten des gewünschten Merkmals der zu untersuchenden Zeitreihe
        comparison_data (pd.DataFrame): Die Vergleichsdaten zu jedem Zeitpunkt der zu untersuchenden Zeitreihe
        comparison_data_options (ComparisonDataExtractionOptions): Optionen bezüglich der Extraktion / des Handlings / der Zugehörigkeit der Vergleichsdaten

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Die beiden Dataframes, die die Daten der genannten Iteratoren enthalten.
    """
    relevant_data_name_and_beginning_timestep = comparison_data.index.names

    # Umbenennen der Indizes, um nachträglich auf Erfassungsbeginn explizit zugreifen zu können
    comparison_data.index.names = ['Zeitreihenname', 'Erfassungsbeginn']

    # Zuordnung der Vergleichszeitreihen zu den Iteratoren für die Objektivität
    mask_same_iterator_relevant_dataseries = mask_for_timestep_query(
        comparison_data, comparison_data_options.period_limitations_same_dataset)
    mask_other_iterator = mask_for_timestep_query(
        comparison_data, comparison_data_options.period_limitations_additional_dataset)

    # Sind keine Zeiträume angegeben, dann handelt es sich nur um einen Datensatz, der gegen die relevante Zeitreihe getestet wird.
    if len(comparison_data_options.period_limitations_same_dataset) == 0:
        if len(comparison_data_options.period_limitations_additional_dataset) > 0:
            raise ValueError("Es wurde keine explizite Zeitspanne für den Datensatz der Vergleichszeitreihe angegeben, damit gehören alle Daten zu diesem. Eine Angabe eines zusätzlichen Datensatzes ist daher nicht gestattet.")
        mask_same_iterator_relevant_dataseries[:] = True

    # Bei mehreren Datensätzen wird im same_iterator am Ende der gesamte direkte Vergleichsdatensatz enthalten sein und alle restlichen Datensätze im anderen Iterator.
    # Bei nur einem Datensatz wird im same_iterator nur die Vergleichszeitreihe sein und der restliche Datensatzteil im anderen Iterator.
    # Das hinzufügen der Vergleichszeitreihe zum same_iterator erfolgt später.
    same_iterator_relevant_dataseries_df = comparison_data[mask_same_iterator_relevant_dataseries]
    other_iterator_df = comparison_data[mask_other_iterator]

    # Sicherstellen, dass alle Zeitreihen, der Vergleichsdaten nun in einer der beiden Dataframes sind
    sanity_check(comparison_data, mask_same_iterator_relevant_dataseries,
                 mask_other_iterator, same_iterator_relevant_dataseries_df, other_iterator_df)

    # Hinzufügen der Vergleichszeitreihe zu dem Iterator
    same_iterator_df, other_iterator_df, one_dataset_only = add_relevant_dataseries_to_iterator(
        data_series, comparison_data, relevant_data_name_and_beginning_timestep, same_iterator_relevant_dataseries_df, other_iterator_df)

    return same_iterator_df, other_iterator_df, one_dataset_only


def mask_for_timestep_query(df: pd.DataFrame, periods: List[Tuple[str, str]]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for start, end in periods:
        start = pd.to_datetime(start, format='%d.%m.%Y %H:%M:%S')
        end = pd.to_datetime(end, format='%d.%m.%Y %H:%M:%S')
        mask |= (df.index.get_level_values('Erfassungsbeginn') >= start) & (
            df.index.get_level_values('Erfassungsbeginn') <= end)
    return mask


def sanity_check(comparison_data: pd.DataFrame, mask_same_iterator_relevant_dataseries: pd.Series, mask_other_iterator: pd.Series, same_iterator_relevant_dataseries_df: pd.DataFrame, other_iterator_df: pd.DataFrame) -> None:
    if len(same_iterator_relevant_dataseries_df.index) + len(other_iterator_df.index) != len(comparison_data.index):
        raise ValueError(
            "Fundamentaler Fehler, das hätte nicht passieren dürfen.")

    # Prüfen, ob in den Vergleichsdaten fehlerhaft Daten waren, die nicht in einer der Zeitspannen lagen
    df_uebrig = comparison_data[~mask_same_iterator_relevant_dataseries &
                                ~mask_other_iterator]
    if len(df_uebrig.index) != 0:
        raise ValueError(
            "Fundamentaler Fehler. Wieso sind in den Vergleichsdaten Zeitreihen, die nun nichtmehr den Zeiten zugeordnet werden können?")


def add_relevant_dataseries_to_iterator(data_series: pd.Series, comparison_data: pd.DataFrame, relevant_data_name_and_beginning_timestep: List[Any], same_iterator_relevant_dataseries_df: pd.DataFrame, other_iterator_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Hier wird dem Dataframe des same_iterators die Vergleichszeitreihe zugefügt.

    Args:
        data_series (pd.Series): Die Daten des gewünschten Merkmals der zu untersuchenden Zeitreihe
        comparison_data (pd.DataFrame): Die Vergleichsdaten zu jedem Zeitpunkt der zu untersuchenden Zeitreihe
        relevant_data_name_and_beginning_timestep (List[Any]): Name der relevanten Zeitreihe und Beginn der Datenerfassung dieser
        same_iterator_relevant_dataseries_df (pd.DataFrame): Das Dataframe des same_iterators, bevor die Vergleichszeitreihe zugefügt wurde

    Returns:
        pd.DataFrame: Das vollständige Dataframe des same_iterator
    """
    relevant_data = pd.DataFrame([data_series])
    relevant_data.columns = comparison_data.columns
    idx = pd.MultiIndex.from_tuples(
        [(relevant_data_name_and_beginning_timestep)], names=None)
    relevant_data.index = idx

    # Wenn nur ein Datensatz verwendet wird, dann ist other_iterator_df zu diesem Zeitpunkt leer
    if len(other_iterator_df) == 0:
        # Variable, um später einsehen zu können, ob es mehrere Datensätze gab
        one_dataset_only = True

        # Verschieben der Vergleichsdaten zum Anderen Iterator, sodass anschließend nur die relevante Vergleichsreihe im same_iterator steht
        other_iterator_df = deepcopy(same_iterator_relevant_dataseries_df)
        same_iterator_df = pd.concat(
            [relevant_data, same_iterator_relevant_dataseries_df.head(0)])
    else:
        # Sind mehrere Datensätze vorhanden muss dennoch diese Variable genutzt werden
        one_dataset_only = False

        # Außerdem wird bei mehreren Datensätzen die relevante Vergleichsreihe den Daten des same_iterators hinzugefügt
        same_iterator_df = pd.concat(
            [relevant_data, same_iterator_relevant_dataseries_df])

    same_iterator_df.index.names = [
        'Zeitreihenname', 'Erfassungsbeginn']

    return same_iterator_df, other_iterator_df, one_dataset_only
