from reputation_calculations.reputation_options import ReputationOptions


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
    options = ReputationOptions()
    # TODO: Hier sollte eine randomisierte Liste erstellt werden, mit einem bestimmten Random-Seed und einer vorgegebenen Länge n.
    # TODO: Diese randomisierte Liste muss dann den options an der richtigen Stelle zugeordnet werden.
    options.reputation_rating = [0.1, 0.2, 0.3, 0.4, 0.5]

    score = calculate_reputation_score(options)
    print(score)
