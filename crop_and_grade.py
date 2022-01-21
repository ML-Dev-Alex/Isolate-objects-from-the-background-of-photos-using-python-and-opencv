import cv2
from isolate_card import isolate_card
from grade_card import check_similarity

if __name__ == "__main__":
    save = True
    display = False
    folder = 'output'

    for i in range(7):
        image_number = i+1
        image_type = "front"
        card_image = isolate_card(image_number=image_number, image_type=image_type,
                                  image_name=f"card_{image_number}_{image_type}", save=save, display=display,
                                  output_folder=f'{folder}')

        reference = cv2.imread(f'./card-pairs/{image_number}/{image_type}_reference.png')
        card_image = check_similarity(
            card_image, reference, name=f"card_{image_number}_{image_type}", save=save, display=display,
            output_folder=f'{folder}')

        image_type = "back"
        card_image = isolate_card(image_number=image_number, image_type=image_type,
                                  image_name=f"card_{image_number}_{image_type}", save=save, display=display,
                                  output_folder=f'{folder}')

        reference = cv2.imread(f'./card-pairs/{image_number}/{image_type}_reference.png')

        card_image = check_similarity(
            card_image, reference, name=f"card_{image_number}_{image_type}", save=save, display=display,
            output_folder=f'{folder}')
