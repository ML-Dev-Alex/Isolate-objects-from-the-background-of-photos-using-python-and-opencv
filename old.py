# # Blur image to get rid of card details, we only need the general contour of the full card for now
    # blurred_front = cv2.GaussianBlur(front, (9, 9), 0)
    # # Blur again to get rid of even more details.
    # blurred_front = cv2.GaussianBlur(blurred_front, (9, 9), 0)
    
    # if display_images:
    #     display_image(blurred_front, "Blurred")

    # gray = cv2.cvtColor(blurred_front, cv2.COLOR_BGR2GRAY)
    # # gray = cv2.blur(gray, (5,5))
    # # gray = cv2.bilateralFilter(gray, 11, 17, 17) #blur. very CPU intensive.
    # # cv2.imshow("Gray map", gray)

    # edges = cv2.Canny(gray, 30, 120)

    # if display_images:
    #     display_image(edges, "Edges")

    # # Transform the colorspace into hue saturation and value
    # hsv = cv2.cvtColor(blurred_front, cv2.COLOR_BGR2HSV)
    
    # # Set thresholds for value, we'll try to find the darkest part of the image (only works on white backgrounds)
    # lower = np.array([0, 0, 150])
    # upper = np.array([255, 255, 255])
    # mask = cv2.inRange(hsv, lower, upper)

    # if display_images:
    #     display_image(mask, "Mask")

    

    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    # dilated_kernel = cv2.dilate(mask, kernel)

    # if display_images:
    #     display_image(dilated_kernel, "Kernel")

    # # Dilate mask to get rid of gaps.
    # dilated_mask = cv2.dilate(mask, (7, 7))
    # dilated_mask = cv2.dilate(dilated_mask, (7, 7))

    # if display_images:
    #     display_image(dilated_mask, "Dilated")

    # contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # # for contour in contours:
    # contoured_front = front.copy()
    # contours_color = (0, 255, 0)
    # cv2.drawContours(contoured_front, contours, -1, contours_color, 3)

    # if display_images:
    #     display_image(contoured_front, "Contours")

    # # Approximate contours to polygons + get bounding rects and circles
    # contours_poly = [None]*len(contours)
    # boundRect = [None]*len(contours)
    # centers = [None]*len(contours)
    # radius = [None]*len(contours)
    # for i, c in enumerate(contours):
    #     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv2.boundingRect(contours_poly[i])
    #     centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    # drawing = np.zeros((dilated_mask.shape[0], dilated_mask.shape[1], 3), dtype=np.uint8)

    # # Draw polygonal contour + bonding rects + circles
    # box_color = (0, 0, 255)
    # for i in range(len(contours)):
    #     cv2.drawContours(drawing, contours_poly, i, box_color)
    #     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #       (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), box_color, 2)
    #     cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), box_color, 2)
    
    
    # if display_images:
    #     display_image(drawing, "Bounding Box")



    # --------

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
