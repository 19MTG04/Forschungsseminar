from dataclasses import dataclass


@dataclass
class ComparisonDataExtractionOptions:
    # Anzahl der Zeitschritte die zeitgleich bei der Suche nach Vergleichsdaten genutzt werden
    window_size: int = 200

    # Entscheidung, ob die Vergleichsdaten nur aus Daten bestehen, deren durchschnittlicher Zeitunterschied in der Messwertaufnahme
    # 10% der Referenz nicht überschreitet. Daten mit doppelter oder halber Frequenz (je +-10 %) sind ebenfalls ähnlich.
    use_data_with_similar_frequency_only: bool = True
