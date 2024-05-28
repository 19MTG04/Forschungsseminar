from reputation_calculations.reputation_options import ReputationOptions, create_reputation_options
import numpy as np


def calculate_reputation_score(options: ReputationOptions) -> float:
    x = options.reputation_rating
    alpha = options.exponential_smoothing_factor

    # Initialisierung der Liste für die Ergebnisse
    all_scores = [0.] * len(x)

    # Der erste Wert wird mit alpha gewichtet
    if len(x) > 0:
        all_scores[0] = alpha * x[0]

        # Rekursive Berechnung für die restlichen Werte
        for i in range(1, len(x)):
            all_scores[i] = (1 - alpha) * all_scores[i - 1] + alpha * x[i]

        reputation_score = all_scores[-1]

    else:
        reputation_score = 0

    return reputation_score


if __name__ == '__main__':
    options = create_reputation_options()

    score = calculate_reputation_score(options)
    print(score)
