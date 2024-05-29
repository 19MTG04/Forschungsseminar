import numpy as np

from reputation_calculations.reputation_options import create_reputation_options

from base_library.community_score import calculate_community_score


def calculate_reputation_score(ratings: np.ndarray, smoothing_factor: float) -> float:
    reputation_score = calculate_community_score(ratings, smoothing_factor)

    return reputation_score


if __name__ == '__main__':
    reputation_options = create_reputation_options()

    score = calculate_reputation_score(
        reputation_options.reputation_rating, reputation_options.exponential_smoothing_factor)
    print(score)
