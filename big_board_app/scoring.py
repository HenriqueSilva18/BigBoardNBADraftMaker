from .config import EVAL_CATEGORIES, WEIGHTS


def calculate_weighted_average(scores):
    total_score = sum(scores[category] * WEIGHTS[category] for category in EVAL_CATEGORIES)
    return round(total_score, 2)


def get_tier(score):
    if score >= 9.5:
        return "Tier 0 - All-Time Talent"
    if score >= 8.5:
        return "Tier 1 - Superstar"
    if score >= 7.5:
        return "Tier 2 - Potential All-NBA"
    if score >= 6.5:
        return "Tier 3 - Potential All-Star"
    if score >= 5:
        return "Tier 4 - Starter"
    return "Tier 5 - Fringe NBA / G-League"
