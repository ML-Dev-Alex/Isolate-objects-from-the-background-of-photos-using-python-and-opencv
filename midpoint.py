def midpoint(ptA, ptB):
    """
    Simple support function that finds the arithmetic mean between two points.
    :param ptA: First point.
    :param ptB: Second point.
    :return: Midpoint.
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
