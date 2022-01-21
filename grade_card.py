import cv2
import imutils
import numpy as np
from show_image import show_image
from skimage.metrics import structural_similarity


def check_similarity(card, reference, name="Card", save=False, display=False, output_folder='output/grading'):
    MAX_FEATURES = 500
    height, width, _ = reference.shape
    resized_card = cv2.resize(card.copy(), (width, height))

    show_image(reference, "Reference", name, save, output_folder, display)

    gray_card = cv2.cvtColor(resized_card, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_card, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_reference, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score.
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove bad matches.
    similar_regions = [i for i in matches if i.distance < 50]
    print(f"Orb Score: {len(similar_regions)/len(matches)}.")
    matches = similar_regions

    # Draw best matches.
    matches_image = cv2.drawMatches(
        resized_card.copy(), keypoints1, reference.copy(), keypoints2, matches, None)
    show_image(matches_image, "Matches", name, save, output_folder, display)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    warped_card = cv2.warpPerspective(
        resized_card, homography, (width, height))
    show_image(warped_card, "Warped Card", name, save, output_folder, display)

    gray_card = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM), ensuring that the difference is returned
    (score, diff) = structural_similarity(gray_card,
                                          gray_reference, full=True, gaussian_weights=True)
    diff = (diff * 255).astype("uint8")
    show_image(diff, "Difference", name, save, output_folder, display)
    print(f"SSIM: {score:.4f}.\n")

    thresh = cv2.threshold(
        diff, 0, 128, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    show_image(thresh, "Thresh", name, save, output_folder, display)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    number_of_diferences = 0
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = w*h
        if area < ((card.shape[0] * card.shape[1]) / 10):
            continue
        number_of_diferences += 1
        cv2.rectangle(card, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_image(card, "Damage", name, save, output_folder, display)

    with open(f'{output_folder}/{name}/Grading.txt', 'w') as f:
        f.write(
            f'{name}:\nSSIM: {score:.4f}\nOrb score: {len(similar_regions)/len(matches)}.')


if __name__ == "__main__":
    card = cv2.imread(f'./output/card_3_front/13_card.jpg')
    reference = cv2.imread(f'./card-pairs/3/front_reference.png')
    card_image = check_similarity(
        card, reference, name="Front Wooper", save=False, display=True)

    card = cv2.imread(f'./output/card_3_back/13_card.jpg')
    reference = cv2.imread(f'./card-pairs/3/back_reference.png')
    card_image = check_similarity(
        card, reference, name="Back Wooper", save=False, display=True)

    card = cv2.imread(f'./output/card_6_front/13_card.jpg')
    reference = cv2.imread(f'./card-pairs/6/front_reference.png')
    card_image = check_similarity(
        card, reference, name="Front Pikachu (damaged)", save=False, display=True)

    card = cv2.imread(f'./output/card_6_back/13_card.jpg')
    reference = cv2.imread(f'./card-pairs/6/back_reference.png')
    card_image = check_similarity(
        card, reference, name="Back Pikachu (damaged)", save=False, display=True)
