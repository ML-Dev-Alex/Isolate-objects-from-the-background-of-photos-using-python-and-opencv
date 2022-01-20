import numpy as np
import cv2
import math
from show_image import show_image


def crop_rectangle(img, rect, padding_multiplier=1.08):
    """
    Crops a rectangle on an image and returns the rotated result.
    :param img: Input image to be cropped.
    :param rect: List of 4 points of the rectangle to crop the image.
    :param padding_multiplier: Number to multiply the width and height of the bounding box by.
    :return: Cropped and rotated image.
    """

    # Get width and height of the detected rectangle.
    box = cv2.boxPoints(rect)
    # box = np.int0(box)
    width = int(rect[1][0]*padding_multiplier)
    height = int(rect[1][1]*padding_multiplier)

    src_pts = box.astype("float32")
    # Coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, (height-1)],
                        [0, 0],
                        [(width-1), 0],
                        [(width-1), (height-1)]], dtype="float32")

    # The perspective transformation matrix.
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Directly warp the rotated rectangle to get the straightened rectangle.
    return cv2.warpPerspective(img, M, (width, height))
