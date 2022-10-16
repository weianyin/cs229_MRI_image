import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage.transform import resize

NUM_HISTOGRAM_BINS = 256
IMAGE_SIZE = (256, 256)


class ImageProcessingOption:
    Flatten = 0
    Histogram = 1


def load_image(filepath):
    return skimage.io.imread(fname=filepath, as_gray=True)


def crop(image):
    """Crop the black borders around the grayscale MRI image.
    We remove the values less than a threshold (instead of zeros)."""
    threshold = 0.25
    xs = np.max(image, axis=0)
    xs = np.nonzero(xs >= threshold)[0]
    xmin, xmax = xs[0], xs[-1]
    ys = np.max(image, axis=1)
    ys = np.nonzero(ys >= threshold)[0]
    ymin, ymax = ys[0], ys[-1]
    return image[ymin:ymax, xmin:xmax]


def crop_and_resize(image):
    image = crop(image)
    image = resize(image, IMAGE_SIZE)
    return image


def load_dataset(split, directory="", show=False, option=ImageProcessingOption.Histogram):
    """
    :param show: If True, displays the images with matplotlib.
    :param option: Image processing option.
    :return:
    """
    assert split in ["train", "test"]
    directory = "/Users/suxuan/PycharmProjects/CS229_Project/archive"

    # Loads the processed dataset if it exists
    numpy_filepath = os.path.join(directory, f"{split}_{option}.npz")
    if os.path.exists(numpy_filepath):
        data = np.load(numpy_filepath)
        return data["X"], data["y"]

    split_folder = "Training" if split == "train" else "Testing"
    folders = [
        "no_tumor",
        "glioma_tumor",
        "meningioma_tumor",
        "pituitary_tumor"
    ]

    if option == ImageProcessingOption.Histogram:
        X = np.empty((0, NUM_HISTOGRAM_BINS))
        y = np.empty((0,))
    else:
        X = np.empty((0, IMAGE_SIZE[0] * IMAGE_SIZE[1]))
        y = np.empty((0,))

    for y_label, folder in enumerate(folders):
        folder_path = os.path.join(directory, split_folder, folder)
        files = os.listdir(folder_path)
        for image_file in files:
            image_filepath = os.path.join(folder_path, image_file)
            image = load_image(image_filepath)
            image = crop_and_resize(image)
            if show:
                plt.imshow(image, cmap="gray")
                plt.show()

            if option == ImageProcessingOption.Histogram:
                # "image": counts of the histogram bins.
                image, bin_edges = np.histogram(image, bins=NUM_HISTOGRAM_BINS, range=(0, 1))
            image = image.reshape((1, -1))
            X = np.concatenate([X, image], axis=0)
            y = np.concatenate([y, np.array([y_label])], axis=0)

    np.savez(numpy_filepath, X=X, y=y)
    return X, y


def load_datasets(directory="", option=ImageProcessingOption.Histogram):
    X_train, y_train = load_dataset(split="train", directory=directory, option=option)
    X_test, y_test = load_dataset(split="test", directory=directory, option=option)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, y_train = load_dataset(split="train")
    print(X_train.shape, y_train.shape)
