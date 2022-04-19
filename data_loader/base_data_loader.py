from utils.image import (
    load_image,
    get_image_paths,
)
import os
from paths import PATHS


class BaseDataLoader(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self._load_atlas_plates()
        self.train_images = None

    def get_train_data(self):
        raise NotImplementedError

    def augment_data(self, images):
        raise NotImplementedError

    def _load_atlas_plates(self):
        print("Loading Mouse Brain Atlas atlas images...")
        self.atlas_images = self._get_images(PATHS.ATLAS_PATH)
        total_images = len(list(self.atlas_images.keys()))
        if total_images > 0:
            print("Loaded. Number of images: ", total_images)
        else:
            raise Exception("No images were found in dir ", PATHS.ATLAS_PATH)

    def _load_training_dataset(self):
        print("Loading training dataset...")
        self.train_images = self._get_images_list(PATHS.TRAIN_PATH)
        total_images = len(list(self.train_images.keys()))
        if total_images > 0:
            print("Loaded. Number of images: ", total_images)
        else:
            raise Exception("No images were found in dir ", PATHS.TRAIN_PATH)

    def _get_images(self, dir: str):
        """
        Load and process (pad, resize) images
        :param dir: Path to the directory where images are located
        :return: dict[slice_number] = image
        """
        img_dict = {}

        for path in get_image_paths(dir):
            slice_no = os.path.basename(path).split(".")[0]
            img_dict[slice_no] = load_image(path, input_shape=self.input_shape)

        return img_dict


    def _get_images_list(self, dir: str):
        """
        Load and process (pad, resize) images
        :param dir: Path to the directory where images are located
        :return: dict[slice_number] = [images]
        """
        img_dict = {}

        for path in get_image_paths(dir):
            slice_no = os.path.basename(path).split('.')[0]
            if '_' in slice_no:
                slice_no = slice_no.split('_')[0]
            if slice_no in img_dict:
                img_dict[slice_no].append(load_image(path, input_shape=self.input_shape))
            else:
                img_dict[slice_no] = [load_image(path, input_shape=self.input_shape)]

        return img_dict