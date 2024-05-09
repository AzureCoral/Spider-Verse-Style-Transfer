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

    for folder in folders:
        universe_data = os.path.join(all_data_folder, folder)
        universe_class = mapping[folder]

        for file in tqdm(os.listdir(universe_data)):
            if file != ".DS_Store":
                file_path = os.path.join(universe_data, file)

                img = tf.io.read_file(file_path)
                img = tf.image.decode_image(img, channels=3)
                img = tf.cast(img, tf.float32)

                img = tf.image.resize(img, [224,224])

                images.append(img)
                labels.append(universe_class)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

def create_mapping(all_data_folder: str) -> Dict[str, int]:
    assert NUM_CLASSES == len(os.listdir(all_data_folder))

    mapping = {}

    one_hot_vectors = tf.one_hot(tf.range(NUM_CLASSES), NUM_CLASSES)

    for folder, vec in zip(os.listdir(all_data_folder), one_hot_vectors):
        mapping[folder] = vec

    assert len(mapping) == NUM_CLASSES

    return mapping

def create_augmentations(all_data_folder: str) -> None:
    for folder in os.listdir(all_data_folder):
        if folder != ".DS_Store":
            universe_data = os.path.join(all_data_folder, folder)
            len_data = len(os.listdir(universe_data))

            for i in range(len_data):
                file = os.listdir(universe_data)[i]
                if file != ".DS_Store":
                    file_path = os.path.join(universe_data, file)

                    img = tf.io.read_file(file_path)

                    img = tf.image.decode_image(img, channels=3)

                    img = tf.cast(img, tf.float32)/255.0

                    write_image(augment_image(img), f"{universe_data}/frame_{len(os.listdir(universe_data))}")
                    write_image(augment_image(img), f"{universe_data}/frame_{len(os.listdir(universe_data))}")

                
def random_crop(image):
    height, width, _ = image.shape

    random_num = tf.random.uniform([], 0, 1)
    cropped_image = tf.cond(random_num < 0.7, lambda: image, lambda: tf.image.random_crop(image, [int(height*0.8), int(width*0.8), 3]))
    return cropped_image

def flip_image_horizontal(image):
    return tf.image.random_flip_left_right(image)

def random_rotation(image):
    random_num = tf.random.uniform([], 0, 1)
    rotated_image = tf.cond(random_num < 0.7, lambda: image, lambda: tf.image.rot90(image))
    return rotated_image

def augment_image(image):
    image = random_crop(image)
    image = flip_image_horizontal(image)
    image = random_rotation(image)
    return image

def write_image(image, path):
    new_img = tf.cast(image*255.0, tf.uint8)

    encoded_image = tf.io.encode_jpeg(new_img)
    tf.io.write_file(path, encoded_image)

if __name__ == "__main__":
    old_length = [len(os.listdir(f"data/{folder}")) for folder in os.listdir("data")]
    create_augmentations("data")
    new_length = [len(os.listdir(f"data/{folder}")) for folder in os.listdir("data")]
    print(old_length)
    print(new_length)
