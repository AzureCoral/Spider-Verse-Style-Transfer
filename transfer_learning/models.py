import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D,\
    Flatten, Dense, Dropout
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

LEARNING_RATE = 0.001
IMAGE_SIZE = [1280, 536, 3]
NUM_CLASSES = 5
NUM_EPOCHS = 10

class TransferCNN(tf.keras.Model):
    def __init__(self) -> None:
        super(TransferCNN, self).__init__()

        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        self.vgg = VGG19(include_top=False, input_size=IMAGE_SIZE, weights='imagenet')

        self.dense_layers = [
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
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

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.loss_fn(y, y_pred)
            print(f"Loss: {loss}\n")
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def train(self, x, y):
        indices = tf.range(tf.shape(x)[0])
        tf.random.shuffle(indices)

        shuffled_inputs = tf.gather(x, indices)
        shuffled_labels = tf.gather(y, indices)

        for i in range(NUM_EPOCHS):
            print(f"Epoch {i+1}/{NUM_EPOCHS}\n")
            self.train_step(shuffled_inputs, shuffled_labels)
