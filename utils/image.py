from typing import List
from skimage import io, img_as_ubyte, exposure
import numpy as np
import tensorflow as tf
import glob


def read_image(path):
    """
    Read image
    :param path: path to image
    :return: image
    """
    image = io.imread(path, as_gray=True)
    image = img_as_ubyte(image)
    return image


def get_padded_image(image):
    """
    Pad image to square shape
    :param image: image (numpy array)
    :return: padded image
    """
    # compute center offset
    h, w = image.shape
    size = max(h, w)
    xx = (size - w) // 2
    yy = (size - h) // 2
    # pad image
    padded_image = np.full((size, size), 0, dtype=np.uint8)
    padded_image[yy : yy + h, xx : xx + w] = image
    # return image
    return padded_image


def load_image(image_path: str, input_shape, pad_image=True):
    """
    Read, pad and resize image. Convert to RGB if needed
    :param image_path: path to the image
    :param input_shape: input shape, e.g. (224,224,3)
    :param pad_image: true if to pad image
    :return: image (numpy array)
    """
    # read image
    img = read_image(image_path)
    # pad image
    if pad_image:
        img = get_padded_image(img)

    img = np.expand_dims(img, axis=-1)

    # resize image
    if img.shape[0] != input_shape[0] or img.shape[1] != input_shape[1]:
        img = tf.image.resize(
            tf.convert_to_tensor(img), (input_shape[0], input_shape[1])
        ).numpy()

    # convert from grayscale to RGB
    if input_shape[2] == 3 and img.shape[2] == 1:
        img = np.concatenate((img,) * 3, axis=-1)
        if img.shape[2] != 3:
            raise Exception(f"3 channels are expected, found {img.shape[2]}")

    return img.astype(np.uint8)


def get_image_paths(directory: str) -> List[str]:
    supported_files = (".tif", ".jpg", ".png")
    return [
        str(file)
        for file in glob.glob(f"{directory}/*.*")
        if str(file).endswith(supported_files)
    ]
