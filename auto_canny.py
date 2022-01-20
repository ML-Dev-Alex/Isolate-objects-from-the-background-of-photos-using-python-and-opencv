import cv2
import numpy as np


def auto_canny(image, sigma=0.33, lower_default=50, higher_default=255):
    """
    Automatically finds the best params to detect edges on a gray image based on the median values of its pixels.
    :param image: Image to be processed.
    :param sigma: Hyper-parameter to determine how open or closed the threshold should be.
    (The lower the sigma, the higher the range).
    :param lower_default: Lower bound of threshold.
    :return: Image containing the edges of the original image.
    """
    # Compute the median of the single channel pixel intensities
    median = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(lower_default, (1.0 - sigma) * median))
    upper = int(min(higher_default, (1.0 + sigma) * median))

    edged = cv2.Canny(image, lower, upper)

    return edged
