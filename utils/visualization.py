import os, sys

sys.path.insert(0, os.getcwd())

import random
from matplotlib import pyplot as plt
from typing import List
from data_loader.base_data_loader import BaseDataLoader
from data_loader.data_loader import TripletDataLoader
from utils.helper import create_folder_if_not_exists


def save_augmented_images(data_loader: BaseDataLoader):
    data_loader._load_training_dataset()
    output_folder = "output/augmented/"
    create_folder_if_not_exists(output_folder)

    rows, cols = 1, 5
    images = []
    for i in range(rows * cols):
        no = random.choice(list(data_loader.train_images.keys()))
        augmented_image = data_loader.augment_data([data_loader.train_images[no]])[0]
        images.append(augmented_image)
        # augmented_image = Image.fromarray(augmented_image)
        # augmented_image.save(os.path.join(output_folder, f"{no}.jpg"))

    fig = plt.figure(figsize=(12, 12))
    for i in range(cols * rows):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")

    plt.savefig(
        os.path.join(output_folder, "augmented_images.png"), bbox_inches="tight"
    )


def show_atlas_and_train_images(data_loader: BaseDataLoader):
    data_loader._load_training_dataset()
    output_folder = "output/figures/"
    create_folder_if_not_exists(output_folder)

    rows, cols = 3, 3
    no_list = [
        random.choice(list(data_loader.train_images.keys())) for _ in range(rows * cols)
    ]

    for dataset in ["atlas", "train"]:

        if dataset == "atlas":
            images = [data_loader.atlas_images[no] for no in no_list]
        elif dataset == "train":
            images = [data_loader.train_images[no] for no in no_list]
        else:
            raise Exception("Wrong dataset!")

        fig = plt.figure(figsize=(12, 12))
        for i in range(cols * rows):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")

        plt.savefig(
            os.path.join(output_folder, f"{dataset}_images.png"), bbox_inches="tight"
        )


def save_top_n_predictions(
    brain_slice_path: str,
    brain_slice_img,
    predicted_plates: List[str],
    data_loader: BaseDataLoader,
):
    cols = 5
    rows = 1
    images = [
        (brain_slice_img, brain_slice_path),
        (data_loader.atlas_images[brain_slice_path], brain_slice_path),
    ]
    images += [
        (data_loader.atlas_images[atlas_path], atlas_path)
        for atlas_path in predicted_plates[:5]
    ]

    fig = plt.figure(figsize=(12, 12))
    for i in range(cols * rows):
        ax = fig.add_subplot(rows, cols, i + 1)
        position = images[i][1]
        if i == 0:
            ax.title.set_text(f"Brain slice: {position}")
        elif i == 1:
            ax.title.set_text(f"Actual atlas plate: {position}")
        else:
            ax.title.set_text(
                f"Predicted plate: {position}",
            )
        ax.title.set_size(9)
        plt.imshow(images[i][0], cmap="gray")
        plt.axis("off")

    output_folder = "output/predictions/"
    create_folder_if_not_exists(output_folder)
    plt.savefig(
        os.path.join(output_folder, f"{brain_slice_path}.png"), bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    data_loader = TripletDataLoader(input_shape=(224, 224, 1), batch_size=16)
    save_augmented_images(data_loader)
    # show_atlas_and_train_images
