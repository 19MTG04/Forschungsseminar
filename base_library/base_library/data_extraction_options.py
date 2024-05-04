from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ComparisonDataExtractionOptions:
    # Anzahl der Zeitschritte die zeitgleich bei der Suche nach Vergleichsdaten genutzt werden
    window_size: int = 100

    # Entscheidung, ob die Vergleichsdaten nur aus Daten bestehen, deren durchschnittlicher Zeitunterschied in der Messwertaufnahme
    # 10% der Referenz nicht überschreitet. Daten mit doppelter oder halber Frequenz (je +-10 %) sind ebenfalls ähnlich.
    use_data_with_similar_frequency_only: bool = True

    # Durch Angabe von [('30.09.2023 12:00:00', '02.10.2023 12:00:00'), ('04.10.2023 12:00:00', '06.10.2023 12:00:00')]
    # werden nur Zeitreihen, die zwischen diesen Daten aufgenommen wurden, als Vergleichsdaten gewertet.
    # Die Anzahl der Angaben ist unbegrenzt.
    period_limitations: List[Tuple] = field(default_factory=list)
