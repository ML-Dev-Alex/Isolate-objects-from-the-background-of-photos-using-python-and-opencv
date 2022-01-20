import cv2
import imutils
import numpy as np
from imutils import perspective
from show_image import show_image, ResizeWithAspectRatio
from auto_canny import auto_canny
from midpoint import midpoint
from crop_rectangle import crop_rectangle


def isolate_card(image_number=1, image_name="Card", image_type="front", save=False, display=False, folder='card-pairs', format='png'):
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
    img = cv2.imread(f'./{folder}/{image_number}/{image_type}.{format}')

    # Check if image is vertical or horizontal, rotate it, and resize to default size mainainting aspect ratio.
    (h, w) = img.shape[:2]
    if h > w:
        # by 90 degrees clockwise
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = ResizeWithAspectRatio(img, width=1920)

    show_image(img, f'{processing_counter}_{image_name}',
               image_name, save=save, display=display)
    processing_counter += 1

    # We begin by pre-processing the image before we try to find the card's contour.

    # Turn it into grayscale and controll the contrast. We don't need color information to find edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contrast+brightness tuning might be important to make some types of images pop, but that's not the case for this one, leave at the default values.
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    show_image(
        contrast, f'{processing_counter}_contrast', image_name, save=save, display=display)
    processing_counter += 1

    # Blur the image. This lowers the complexity of the image and facilitates edge detecting.
    blur = cv2.bilateralFilter(contrast, 9, 150, 150)
    show_image(blur, f'{processing_counter}_blur',
               image_name, save=save, display=display)
    processing_counter += 1


    # Binarize the image, this allows for better edge detection.
    # Compute the median of the single channel pixel intensities, to automatically find the best threshold values.
    median = np.median(blur)
    # Sigma adjusts how confident well you believe the automatic thresholding will be better than the manually selected values.
    # the higher the value, the more likely it is to be selected.
    sigma = .2
    lower = int(max(40, (1.0 - sigma) * median))
    # We only want to select the minimum value because we are assuming the background will be white. This skews the mean one way, and guarantees
    # the card will be darker.
    # upper = int(min(255, (1.0 + sigma) * median)) 
    thresh = cv2.threshold(blur, lower, 255, cv2.THRESH_BINARY)[1]
    if display:
        show_image(thresh, f'{processing_counter}_thresh',
                   image_name, save=save)
    processing_counter += 1

    # Use the canny algorithm to find edges.
    canny = auto_canny(thresh)
    show_image(canny, f'{processing_counter}_canny',
               image_name, save=save, display=display)
    processing_counter += 1

    # Create a vertical line kernel and dilate edges to increase the chance we'll be able to get the full contourn later.
    kernel = np.array(([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]]), dtype=np.uint8)
    edged = cv2.dilate(canny, kernel, iterations=2)
    show_image(
        edged, f'{processing_counter}_dilated_vertical', image_name, save=save, display=display)
    processing_counter += 1

    # Do the same with a horizontal line kernel.
    kernel = np.array(([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]), dtype=np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    show_image(
        edged, f'{processing_counter}_dilated_horizontal', image_name, save=save, display=display)
    processing_counter += 1

    # One more morphological transformation to close/fill holes on the edges, this time a full closed kernel.
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
    show_image(closed, f'{processing_counter}_closed',
               image_name, save=save, display=display)
    processing_counter += 1

    # Now we can start process of finding the contour
    cnts = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    biggestContours = []
    biggestArea = 0
    for _, contour in enumerate(cnts):
        # If the contour is not sufficiently large, ignore it, otherwise add it to list of biggest contours
        area = cv2.contourArea(contour)
        if area < ((closed.shape[0] * closed.shape[1]) / 10) and len(biggestContours) > 1:
            continue
        else:
            biggestContours.append(contour)

        # Check if current contour has the biggest area in all of the contours, and save it if so.
        if area >= biggestArea:
            biggestArea = area
            biggestContour = contour

    # Draw all of the biggest contours.
    if len(biggestContours) > 0:
        contours_image = cv2.drawContours(
            img.copy(), biggestContours, -1, (255, 0, 0), 10)
        show_image(
            contours_image, f'{processing_counter}_biggest_contours_image', image_name, save=save, display=display)
        processing_counter += 1

        # Calculate hull points for each of the biggest contours and display them. (This tries to close the countours)
        hull = []
        for i in range(len(biggestContours)):
            hull.append(cv2.convexHull(biggestContours[i], False))
            hull_image = cv2.drawContours(img.copy(), hull, -1, (0, 0, 0), -1)
        
        show_image(
            hull_image, f'{processing_counter}_convex_hull_contours_image', image_name, save=save, display=display)
        processing_counter += 1


    # If we found a big contour, draw the biggest one of them on the image and display it.
    if biggestContour is not None:
        contours_image = cv2.drawContours(
            img.copy(), biggestContour, -1, (0, 0, 255), 20)
        show_image(
            contours_image, f'{processing_counter}_biggest_contour_image', image_name, save=save, display=display)
        processing_counter += 1

        # Try another approximation for the bounding box. (Not worth using, our solution works better).
        epsilon = 0.1*cv2.arcLength(biggestContour, True)
        approx = cv2.approxPolyDP(biggestContour, epsilon, True)
        contours_image = cv2.drawContours(
            img.copy(), approx, -1, (0, 255, 0), 10)
        show_image(
            contours_image, f'{processing_counter}_approximate_box_contours_image', image_name, save=save, display=display)
        processing_counter += 1
    else:
        print(
            f"Could not find contourn on image {image_number}_{image_type}. Please take another photo with clear white background and good lighting/shadows.")
        return None

    # Compute the rotated bounding box of the biggest contour
    box = cv2.minAreaRect(biggestContour)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Order the points in the contour such that they appear in
    # top-left, top-right, bottom-right, and bottom-left order,
    # then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(closed, [box.astype("int")], -1, (0, 255, 0), 10)

    # Loop over the original points and draw them on the image
    for (x, y) in box:
        cv2.circle(closed, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Unpack the ordered bounding box, tl = top left, tr = top right, br = bottom right, bl = bottom left
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
    cv2.circle(closed, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(closed, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(closed, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(closed, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # Draw lines between the midpoints
    cv2.line(closed, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(closed, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)
    show_image(
        closed, f'{processing_counter}_boxes', image_name, save=save, display=display)
    processing_counter += 1

    # Crop only the card out of the original image, and display/save the transformed versions of it.
    box = cv2.minAreaRect(biggestContour)
    card = crop_rectangle(img, box)
    # Check if image is vertical or horizontal then rotate it to vertical orientation.
    (h, w) = card.shape[:2]
    if h < w:
        # by 90 degrees clockwise
        card = cv2.rotate(card, cv2.cv2.ROTATE_90_CLOCKWISE)
    show_image(card, f'{processing_counter}_card',
               image_name, save=save, display=display)
    processing_counter += 1

    # Return the cropped card without the background.
    return card


# If we execute this function by itself, it shows us how it works. If you call it , by default, it'll only return the isolated card image.
if __name__ == "__main__":
    save = True
    display = False

    image_number = 1
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 2
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 3
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 4
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 5
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 6
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 7
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_number = 1
    image_type = "front"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)
    image_type = "back"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display)

    image_number = 0
    folder = 'tests'

    image_type = "test_1"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='png')

    image_type = "test_2"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_3"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_4"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_5"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_6"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_7"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_8"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_9"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_10"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_11"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_12"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')

    image_type = "test_13"
    card_image = isolate_card(image_number=image_number, image_type=image_type,
                              image_name=f"card_{image_number}_{image_type}", save=save, display=display, folder=folder, format='jpg')
