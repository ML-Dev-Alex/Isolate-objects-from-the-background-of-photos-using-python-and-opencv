import cv2
import imutils
import numpy as np
from imutils import perspective
from show_image import show_image
from auto_canny import auto_canny
from midpoint import midpoint
from crop_rectangle import crop_rectangle


def isolate_card(image_number=1, image_name="Card", image_type="front", save=False, display=False):
    """
    Isolates a card from it's background.
    :param image_number: The number of the image to be isolated.
    :param image_name: The name of the image.
    :param image_type: The type of the image, can be front or back.
    :param save: If True, will save processing images in the harddrive.
    :param display: If True, display images on screen as processing occurs.
    """
    processing_counter = 0

    # Open image.
    img = cv2.imread(f'./card-pairs/{image_number}/{image_type}.png')
    if display:
        show_image(img, f'{processing_counter}_{image_name}',
                   image_name, save=save)
    processing_counter += 1

    # We begin by pre-processing the image before we try to find the card's contour.

    # Turn it into grayscale and controll the contrast.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    if display:
        show_image(
            contrast, f'{processing_counter}_contrast', image_name, save=save)
    processing_counter += 1

    # Blur the image.
    blur = cv2.bilateralFilter(contrast, 9, 150, 150)
    if display:
        show_image(blur, f'{processing_counter}_blur', image_name, save=save)
    processing_counter += 1

    # Binarize it.
    thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)[1]
    if display:
        show_image(thresh, f'{processing_counter}_thresh',
                   image_name, save=save)
    processing_counter += 1

    # Use the canny algorithm to find edges.
    canny = auto_canny(thresh)
    if display:
        show_image(canny, f'{processing_counter}_canny', image_name, save=save)
    processing_counter += 1

    # Create a vertical line kernel and dilate edges to increase the chance we'll be able to get the full contourn later.
    kernel = np.array(([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]]), dtype=np.uint8)
    edged = cv2.dilate(canny, kernel, iterations=1)
    if display:
        show_image(
            edged, f'{processing_counter}_dilated_vertical', image_name, save=save)
    processing_counter += 1

    # Do the same with a horizontal line kernel.
    kernel = np.array(([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]), dtype=np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    if display:
        show_image(
            edged, f'{processing_counter}_dilated_horizontal', image_name, save=save)
    processing_counter += 1

    # One more morphological transformation to close/fill holes on the edges, this time a full closed kernel.
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
    if display:
        show_image(closed, f'{processing_counter}_closed',
                   image_name, save=save)
    processing_counter += 1

    # Now we can start process of finding the contour
    cnts = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    biggestArea = 0

    # Create a copy of the edged image in order to preserve it for later.
    edged_copy = edged.copy()
    biggestContours = []
    for _, contour in enumerate(cnts):
        # If the contour is not sufficiently large, ignore it, otherwise add it to list of biggest contours
        area = cv2.contourArea(contour)
        if area < ((edged_copy.shape[0] * edged_copy.shape[1]) / 10):
            continue
        else:
            biggestContours.append(contour)

        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear in
        # top-left, top-right, bottom-right, and bottom-left order,
        # then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(edged_copy, [box.astype("int")], -1, (0, 255, 0), 10)

        # Loop over the original points and draw them on the image
        for (x, y) in box:
            cv2.circle(edged_copy, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Unpack the ordered bounding box,
        (tl, tr, br, bl) = box

        # Then compute the midpoint between the top-left and top-right coordinates,
        (tltrX, tltrY) = midpoint(tl, tr)
        # Followed by the midpoint between bottom-left and bottom-right coordinates
        (blbrX, blbrY) = midpoint(bl, br)

        # Compute the midpoint between the top-left and top-right points,
        (tlblX, tlblY) = midpoint(tl, bl)
        # Followed by the midpoint between the top-right and bottom-right
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(edged_copy, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(edged_copy, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(edged_copy, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(edged_copy, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(edged_copy, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(edged_copy, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # Check if current contour has the biggest area in all of the contours, and save it if so.
        if area >= biggestArea:
            biggestArea = area
            biggestContour = contour

        # If we found a big contour, draw the biggest one of them on the image and display it.
        if len(biggestContours) > 0:
            contours_image = cv2.drawContours(
                img.copy(), biggestContours, -1, (255, 0, 0), 10)
            if display:
                show_image(
                    contours_image, f'{processing_counter}_contours_image', image_name, save=save)
            processing_counter += 1
        else:
            print("Could not find contourn on image. Please take another photo with clear white background and good lighting.")

    # Select the bounding box of the biggest contour on the image (hopefully the card).
    box = cv2.minAreaRect(biggestContour)
    if display:
        show_image(
            edged_copy, f'{processing_counter}_boxes', image_name, save=save)
    processing_counter += 1

    # Crop only the card out of the original image, and display/save the transformed versions of it.
    card = crop_rectangle(img, box)
    if display:
        show_image(card, f'{processing_counter}_card', image_name, save=save)
    processing_counter += 1

    # card_gray = crop_rectangle(gray, box)
    # card_contrast = cv2.convertScaleAbs(card_gray, alpha=alpha, beta=beta)
    # if display:
    #     show_image(
    #         card_contrast, f'{processing_counter}_card_contrast', image_name, save=save)
    # processing_counter += 1

    # card_blur = crop_rectangle(blur, box)
    # if display:
    #     show_image(
    #         card_blur, f'{processing_counter}_card_blur', image_name, save=save)
    # processing_counter += 1

    # card_canny = crop_rectangle(canny, box)
    # if display:
    #     show_image(
    #         card_canny, f'{processing_counter}_card_canny', image_name, save=save)
    # processing_counter += 1

    # card_edged = crop_rectangle(edged, box)
    # if display:
    #     show_image(
    #         card_edged, f'{processing_counter}_card_dilated', image_name, save=save)
    # processing_counter += 1

    # mask = np.ones(img.shape[:2], dtype="uint8") * 255
    # cv2.drawContours(mask, [biggestContour], -1, 0, -1)

    # mask = (255 - mask)
    # rotated = cv2.bitwise_and(img, img, mask=mask)
    # cropped = rotated.copy()
    # rotated = crop_rectangle(rotated, box)
    # mask = crop_rectangle(mask, box)

    # if display:
    #     show_image(mask, f'{processing_counter}_card_mask',
    #                image_name, save=save)
    # processing_counter += 1
    # if display:
    #     show_image(rotated, f'{processing_counter}_rotated',
    #                image_name, save=save)
    # processing_counter += 1
    # if display:
    #     show_image(
    #         cropped, f'{processing_counter}_card_cropped', image_name, save=save)
    # processing_counter += 1

    # Return the cropped card without the background
    return card


# If we execute this function by itself, it shows us how it works. If you call it , by default, it'll only return the isolated card image.
if __name__ == "__main__":
    image_number = 1
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=True, display=True)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=True, display=True)
