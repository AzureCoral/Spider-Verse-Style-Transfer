from keras.api.saving import register_keras_serializable
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D,\
    Flatten, Dense, Dropout
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras

LEARNING_RATE = 0.001
IMAGE_SIZE = [224, 224,3]
NUM_CLASSES = 5
NUM_EPOCHS = 10
BATCH_SIZE = 16

@keras.saving.register_keras_serializable()
class TransferCNN(tf.keras.Model):
    def __init__(self, input_shape) -> None:
        super(TransferCNN, self).__init__()

        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        self.vgg = VGG19(include_top=False, input_shape=input_shape, weights='imagenet')

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(256, activation='relu')
        self.dropout2 = Dropout(0.3)
        self.dense3 = Dense(NUM_CLASSES, activation='softmax')

    def call(self, inputs):
        x = self.vgg(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)

    @staticmethod
    def loss_fn(y, y_pred):
        return categorical_crossentropy(y, y_pred)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.loss_fn(y, y_pred)
        loss_value = tf.reduce_mean(loss)
        print(f"Loss: {loss_value.numpy()}\n")
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def train(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x)).batch(BATCH_SIZE)

        for i in range(NUM_EPOCHS):
            print(f"Epoch {i+1}/{NUM_EPOCHS}\n")
            for batch_x, batch_y in dataset:
                self.train_step(batch_x, batch_y)

    def build(self, input_shape):
        self.vgg.build(input_shape)
        super().build(input_shape)
