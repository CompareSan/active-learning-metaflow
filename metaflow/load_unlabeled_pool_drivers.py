import cv2
import os

def get_im_cv2(path, color_type=1):
    """
    Load and resize images using the OpenCV library in Python.

    Args:
        path (str): The path to the image file.
        color_type (int, optional): The color type of the image. It can be either 1 (grayscale) or 3 (RGB). Default is 1.

    Returns:
        numpy array: The resized image.

    Example:
        path = 'path/to/image.jpg'
        color_type = 1
        img = get_im_cv2(path, color_type)
    """
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)

    #print('Original Dimensions : ',img.shape)

    scale_percent = 5 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ', resized.shape)
    return resized

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
    
    for j in range(10):
        print('Loading data...')
        #print(os.getcwd())
        directory = os.path.join('/Users/filippobuoncompagni/renaulution_qai_metaflow/data_drivers',
                                'imgs', 'train', 'c' + str(j))
        
        for file in os.listdir(directory):
            img = get_im_cv2(os.path.join(directory,file), color_type)
            X_train.append(img)
            if j == 0:
                y_train.append(0)
            else:
                y_train.append(1)
    
    return X_train, y_train


