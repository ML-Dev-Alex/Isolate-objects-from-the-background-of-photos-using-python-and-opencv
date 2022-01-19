
import math


def get_angle(a, b, c):
    """
    Finds the angle (in degrees) between three points.
    :param a: First point.
    :param b: Second point.
    :param c: Third point.
    :return: A number between 0 and 360 degrees representing the angle between the three points.
    """
    ang = math.degrees(math.atan2(
        c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang
