import os

import cv2


def get_unlabeled_pool_drivers(color_type=1):
    """
    Loads and resizes images from a directory and assigns labels based on the directory name.

    Args:
        color_type (int, optional): The color type of the image. It can be either 1 (grayscale) or 3 (RGB). Default is 1.

    Returns:
        tuple: A tuple containing two lists: `X_train` containing the resized images and `y_train` containing the corresponding labels.
    """
    X_train = []
    y_train = []
    SCALE_PERCENT = 5
    for j in range(10):
        print("Loading data...")
        directory = os.path.join(
            "/Users/filippobuoncompagni/renaulution_qai_metaflow/data_drivers",
            "imgs",
            "train",
            "c" + str(j),
        )

        for file in os.listdir(directory):
            image = get_image(os.path.join(directory, file), color_type)
            resized_image = resize_image(image, SCALE_PERCENT)
            X_train.append(resized_image)
            if j == 0:
                y_train.append(0)
            else:
                y_train.append(1)

    return X_train, y_train


def get_image(path, color_type=1):
    """
    Load images using the OpenCV library in Python.

    Args:
        path (str): The path to the image file.
        color_type (int, optional): The color type of the image. It can be either 1 (grayscale) or 3 (RGB). Default is 1.

    Returns:
        numpy array: The image.

    Example:
        path = 'path/to/image.jpg'
        color_type = 1
        image = get_image(path, color_type)
    """

    if color_type == 1:
        image = cv2.imread(path, 0)
    elif color_type == 3:
        image = cv2.imread(path)

    return image


def resize_image(image, scale_percent):
    """
    Resize an image using the OpenCV library.

    Args:
        image (numpy array): The input image to be resized.
        scale_percent (int): The percentage by which the image should be scaled.

    Returns:
        numpy array: The resized image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image
