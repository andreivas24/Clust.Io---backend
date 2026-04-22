# Utility function to compute Pearson correlation coefficient
import math

def compute_pearson_correlation(x_values, y_values):
    """
    Returns Pearson correlation coefficient for two numeric lists.
    """
    if not x_values or not y_values or len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    n = len(x_values)

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))

    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))

    if denom_x == 0 or denom_y == 0:
        return None

    return round(numerator / (denom_x * denom_y), 4)