import numpy as np
import cv2
import math


def crop_rectangle(img, rect, cut=True):
    """
    Crops a rectangle on an image and returns the rotated result.
    :param img: Input image to be cropped.
    :param rect: List of 4 points of the rectangle to crop the image.
    :param cut: Boolean to determine whether to actually cut the image before returning, or not.
    :return: Either a cropped image, or the full image with the rectangle drawn into it depending on the cut variable.
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if not cut:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 10)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    mult = 1.0
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))

    if not cut:
        cv2.circle(img, center, 10, (0, 255, 0), -1)

    if not cut:
        size = (img.shape[0] + int(math.ceil(W)),
                img.shape[1] + int(math.ceil(H)))
        center = (int(size[0] / 2), int(size[1] / 2))

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    if cut:
        croppedW = W if not rotated else H
        croppedH = H if not rotated else W
    else:
        croppedW = img.shape[0] if not rotated else img.shape[1]
        croppedH = img.shape[1] if not rotated else img.shape[0]

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)),
                                       (size[0] / 2, size[1] / 2))
    return croppedRotated
