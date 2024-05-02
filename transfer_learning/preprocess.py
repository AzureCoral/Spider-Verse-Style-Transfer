from typing import Dict, Tuple
import cv2
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input

NUM_CLASSES = 5

def get_data(all_data_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the folders from the given directory and processes the images and labels.

    Args:
    all_data_folder (str): The folder containing all the data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The processed images and labels
    """
    images = []
    labels = []

    mapping = create_mapping(all_data_folder)

    for folder in os.listdir(all_data_folder):
        universe_data = os.path.join(all_data_folder, folder)
        universe_class = mapping[folder] 

        for file in tqdm(os.listdir(universe_data)):
            if file != ".DS_Store":
                file_path = os.path.join(universe_data, file)

                img = cv2.imread(file_path)
                img = cv2.resize(img, (224, 224))
                img = preprocess_input(img)

                images.append(img)
                labels.append(universe_class)

    return np.array(images), np.array(labels)


def create_mapping(all_data_folder: str) -> Dict[str, int]:
    assert NUM_CLASSES == len(os.listdir(all_data_folder))

    mapping = {}

    one_hot_vectors = tf.one_hot(tf.range(NUM_CLASSES), NUM_CLASSES)

    for folder, vec in zip(os.listdir(all_data_folder), one_hot_vectors):
        mapping[folder] = vec

    assert len(mapping) == NUM_CLASSES

    return mapping
