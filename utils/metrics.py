import os.path
import numpy as np
import time

from data_loader.base_data_loader import BaseDataLoader
from models.base_model import BaseModel
from tensorflow.keras.models import Model
from utils.image import load_image, get_image_paths
from utils.distance import euclidean_distance_numpy
from utils.visualization import save_top_n_predictions


class Metrics:
    def __init__(
        self, data_loader: BaseDataLoader, model: BaseModel, dataset_path: str
    ):
        self.data_loader = data_loader
        self.model = model
        self.dataset_path = dataset_path

        # load images to memory
        self.test_images = self._read_test_images()
        # get a list of paths of test images
        self.test_images_paths = list(self.test_images.keys())

        self.mae = None
        self.top_n = None

    def _read_test_images(self):
        print(f"Loading test images from dir {self.dataset_path}")
        test_images = {}
        for brain_slice_path in get_image_paths(self.dataset_path):
            brain_slice = load_image(
                brain_slice_path, input_shape=self.data_loader.input_shape
            )
            slice_no = os.path.basename(brain_slice_path).split(".")[0]
            test_images[slice_no] = brain_slice
        img_list_len = len(list(test_images.keys()))
        if img_list_len == 0:
            raise Exception("No images were found in dir: ", self.dataset_path)
        else:
            print(f"Number of test images: {img_list_len}")
        return test_images

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _get_embedding(self, img_dict, model: Model):
        """
        Get embeddings for images in img_dict
        :param img_dict: dict[image_basename] = image
        :param model: model
        :return: dict[image_basename] = embedding
        """
        embeddings = {}
        for chunk in list(self._chunks(list(img_dict.keys()), 33)):
            img_list = []
            for dict_key in chunk:
                img_list.append(img_dict[dict_key])
            embeddings_list = model.predict(np.array(img_list))
            for i in range(len(chunk)):
                embeddings[chunk[i]] = embeddings_list[i]

        return embeddings

    def compute(self, visualize: bool = False):
        """
        Compute TOP N accuracy and MAE (mean absolute error)
        :param model: trained model
        :param visualize: save figures of top n predictions
        :return: MAE
        """
        total_error = 0
        self.top_n_dict = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}

        start_time = time.time()

        print("Computing embeddings for atlas plates...")
        atlas_embeddings = self._get_embedding(
            self.data_loader.atlas_images, self.model.get_model()
        )
        print("Computing embeddings for test images...")
        test_img_embeddings = self._get_embedding(
            self.test_images, self.model.get_model()
        )

        # process all test images
        for brain_slice_path in self.test_images_paths:
            # get embeddings
            brain_embedding = test_img_embeddings[brain_slice_path]
            predicted = {}
            # compute euclidean distance between a test image and each atlas plate
            for atlas_no, embedding in atlas_embeddings.items():
                predicted[atlas_no] = euclidean_distance_numpy(
                    (embedding, brain_embedding)
                )
            # sort based on euclidean distance
            sorted_distance = list(
                dict(
                    sorted(predicted.items(), key=lambda item: item[1], reverse=False)
                ).keys()
            )
            # compute top N accuracy
            for top_n in list(self.top_n_dict.keys()):
                top_sim = sorted_distance[:top_n]
                # if top_n == 5:
                #     print(f"True: {brain_slice_path}, top_5_sim: {str(top_sim)}")
                if brain_slice_path in top_sim:
                    self.top_n_dict[top_n] += 1
            total_error = total_error + sorted_distance.index(brain_slice_path)

            if visualize:
                save_top_n_predictions(
                    brain_slice_path=brain_slice_path,
                    brain_slice_img=self.test_images[brain_slice_path],
                    data_loader=self.data_loader,
                    predicted_plates=sorted_distance,
                )

        self.mae = round(total_error / len(self.test_images_paths), 2)

        print(f"Results (dataset {self.dataset_path})")
        print(f"Execution time: {str(time.time() - start_time)}")
        print("Top N accuracy: ", self.top_n_dict)
        print(f"MAE: {self.mae}")

        return self.mae
