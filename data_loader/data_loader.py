from data_loader.base_data_loader import BaseDataLoader
import numpy as np
import random
import imgaug.augmenters as iaa


class TripletDataLoader(BaseDataLoader):
    def __init__(self, input_shape, augmentation: bool = False, batch_size: int = 16):
        super(TripletDataLoader, self).__init__(input_shape)
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.seq = None

    def get_train_data(self):
        if not self.train_images:
            self._load_training_dataset()
        atlas_file_names = list(self.atlas_images.keys())
        train_file_names = list(self.train_images.keys())

        imgs_per_class = 2
        atlas_train_steps = int(self.batch_size * 0.25 if self.augmentation else 0)
        brain_train_steps = int(self.batch_size * 0.75 if self.augmentation else 1)

        while True:
            images = []
            labels = []

            for i in range(brain_train_steps):
                brain_slice_no = random.choice(train_file_names)
                labels.append(brain_slice_no)
                images.append(random.choice(self.train_images[brain_slice_no]))
                for u in range(imgs_per_class - 1):
                    labels.append(brain_slice_no)
                    images.append(
                        self.augment_data([self.atlas_images[brain_slice_no]])[0]
                    )

            for i in range(atlas_train_steps):
                atlas_no = random.choice(atlas_file_names)
                for u in range(imgs_per_class):
                    labels.append(atlas_no)
                    images.append(self.augment_data([self.atlas_images[atlas_no]])[0])

            yield np.array(images), np.array(labels)

    def augment_data(self, images):
        if not self.seq:
            self.seq = iaa.Sequential(
                [
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.Affine(scale=(1.0, 1.8)),
                    iaa.CropAndPad(percent=(-0.10, 0.10)),
                    iaa.CoarsePepper(0.1, size_percent=(0.01, 0.01)),
                ]
            )
        return self.seq(images=images)
