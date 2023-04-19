"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 312708969


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Load the image using OpenCV
    image = cv2.imread(filename)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to read image from {filename}")

    if representation == LOAD_RGB:
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to float and normalize the intensities to [0, 1]
    image_np = image.astype(np.float32) / 255.0

    return image_np


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    image_np = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(image_np , cmap='gray')
    else:
        plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transform_mat = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])

    imgYIQ = np.dot(imgRGB, transform_mat.T)

    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # YIQ to RGB linear transformation matrix
    transform_mat = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.106, 1.703]])

    # Perform matrix multiplication to convert YIQ to RGB
    imgRGB = np.dot(imgYIQ, transform_mat.T)

    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """

    imOrig = (imgOrig * 255).astype(np.uint8)

    # Calculate original image histogram
    histOrg, _ = np.histogram(imOrig.flatten(), bins=256, range=(0, 255))

    # Calculate the normalized
    cumSum = histOrg.cumsum()
    cumSumNorm = cumSum / cumSum[-1]

    # Create the LookUpTable (LUT)
    LUT = (cumSumNorm * 255).astype(np.uint8)

    # Replace each intensity
    imEq = LUT[imOrig]

    # Calculate histogram of the equalized image
    histEq, _ = np.histogram(imEq.flatten(), bins=256, range=(0, 255))
    imEq = imEq.astype(np.float32) / 255

    return imEq, histOrg, histEq



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    img = np.copy(imOrig)
    if img.ndim == 3:
        imgYIQ = transformRGB2YIQ(imOrig)
        imOrig = np.copy(imgYIQ[:, :, 0])
    else:  # Case grayscale
        imgYIQ = imOrig

    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    imOrig = imOrig.astype('uint8')
    hist, bins = np.histogram(imOrig, 256, [0, 255])

    # Find boundaries
    z_array = np.zeros(nQuant + 1, dtype=int)  # z is represents the boundaries
    for i in range(1, nQuant):
        z_array[i] = z_array[i - 1] + int(255 / nQuant)
    z_array[nQuant] = 255
    q_array = np.zeros(nQuant)

    images_list = list()
    mse_list = list()

    for i in range(nIter):
        new_image = np.zeros(imOrig.shape)
        for j in range(len(q_array)):
            if j == len(q_array) - 1:
                right = z_array[j + 1] + 1
            else:
                right = z_array[j + 1]
            range_cell = np.arange(z_array[j], right)
            q_array[j] = np.average(range_cell, weights=hist[z_array[j]:right])
            mat = np.logical_and(imOrig >= z_array[j], imOrig < right)
            new_image[mat] = q_array[j]

        mse_list.append(np.sum(np.square(np.subtract(new_image, imOrig))) / imOrig.size)

        if img.ndim == 3:
            imgYIQ[:, :, 0] = new_image / 255
            new_image = transformYIQ2RGB(imgYIQ)

        images_list.append(new_image)

        for bd in range(1, len(z_array) - 1):
            z_array[bd] = (q_array[bd - 1] + q_array[bd]) / 2

        if len(mse_list) >= 2:
            if np.abs(mse_list[-1] - mse_list[-2]) <= 0.000001:
                break

    return images_list, mse_list