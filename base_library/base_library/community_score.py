import numpy as np


def calculate_community_score(ratings: np.ndarray, smoothing_factor: float) -> float:
    # Initialisierung der Liste für die Ergebnisse
    all_scores = [0.] * len(ratings)

    # Der erste Wert wird mit alpha gewichtet
    if len(ratings) > 0:
        all_scores[0] = smoothing_factor * ratings[0]

        # Rekursive Berechnung für die restlichen Werte
        for i in range(1, len(ratings)):
            all_scores[i] = (1 - smoothing_factor) * \
                all_scores[i - 1] + smoothing_factor * ratings[i]

        reputation_score = all_scores[-1]

    else:
        reputation_score = 0

    return reputation_score
