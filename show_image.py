import cv2
import pyautogui
import os

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resized image to be displayed, respecting it's aspect ratio.
    :param image: Image to be resized.
    :width: Desired width.
    :height: Desired height. If both are passed, defaults to desired width.
    :inter: interpolation type.
    """
    dim = None
    (h, w) = image.shape[:2]

    # Must pass either a width or a height as a parameter, otherwise, return the image without resizing it.
    if width is None and height is None:
        return image

    # Resize image, preserving aspect ratio based on width or height, depending on which was passed.
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def show_image(img, title='image', image_name='image', save=True, directory='output', display=True):
    """
    Displays an image on screen and saves it on the hard-drive if desired.
    :param img: List of cv2 images to display and save.
    :param title: Image title, generally a description of what kind of transformation was applied to the image.
    :param image_name: Name of the current image being displayed, a new folder will be created for the image name supplied.
    :param save: If True, saves image on harddrive.
    :param directory: Output directory to save image in.
    """

    # Get screen resolution.
    screen_width, screen_height = pyautogui.size()

    if save:
        # Create output folders if they do not exist.
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not os.path.isdir(f'{directory}/{image_name}'):
            os.mkdir(f'{directory}/{image_name}')

    
    if display:
        # Initialize display window
        window_name = f'{title}'
        cv2.namedWindow(window_name)

        # Resize if image is too big.
        resized = False
        (h, w) = img.shape[:2]
        # If too wide, resize with respect to width.
        if (w > screen_width/3):
            resized = True
            resized_img = ResizeWithAspectRatio(img, width=int(screen_width/2))
            (resized_h, resized_w) = resized_img.shape[:2]
            # Center window.
            cv2.moveWindow(window_name, int(screen_width/2 -
                            resized_w/2), int(screen_height/2 - resized_h/2))
        # If too tall, resize with respect to height.
        elif(h > screen_height/3):
            resized = True
            resized_img = ResizeWithAspectRatio(img, height=int(screen_height/2))
            (resized_h, resized_w) = resized_img.shape[:2]
            # Center window.
            cv2.moveWindow(window_name, int(screen_width/2 -
                            resized_w/2), int(screen_height/2 - resized_h/2))
        # Otherwise display original image.
        else:
            # Center window.
            cv2.moveWindow(window_name, int(screen_width/2 -
                            w/2), int(screen_height/2 - h/2))   
    
        # And finaly display it on screen.
        if resized:
            cv2.imshow(f'{title}', resized_img)
        else:
            cv2.imshow(f'{title}', img)

        # Wait for input before closing image and moving on.
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save image on destination folder.
    if save:
        cv2.imwrite(f'{directory}/{image_name}/{title}.jpg', img)

    
