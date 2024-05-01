import tensorflow as t
import numpy as np
import os
from preprocess import *

from keras.layers import Conv2D, MaxPool2D,\
    Flatten, Dense, Dropout, Sequential

from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

LEARNING_RATE = 0.001
IMAGE_SIZE = [1280, 536, 3]
NUM_CLASSES = 5

class TransferCNN(tf.keras.Model):
    def __init__(self) -> None:
        super(TransferCNN, self).__init__()

        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        self.vgg = VGG19(include_top=False, input_size=IMAGE_SIZE, weights='imagenet')

        self.dense_layers = [
            Flatten(),
            Dense(256, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ]

        self.dense_layers = Sequential(self.dense_layers)

        self.model = Sequential([
            self.vgg,
            self.dense_layers
        ])

    def call(self, x):
        return self.model(x)

    @staticmethod
    def loss_fn(y, y_pred):
        return categorical_crossentropy(y, y_pred)

