from preprocess import get_data
from models import TransferCNN
import tensorflow as tf
import numpy as np

CHECKPOINT_PATH = "./checkpoints"
IMAGE_SIZE = (224,224,3)

def main():
    all_data_folder = "data"
    print("Starting Preprocessing...")
    images, labels = get_data(all_data_folder)
    print("Finished Preprocessing!")print("Starting Training without Weights")

    model = TransferCNN(load=False, nontrainable_layers=None)

    model.build(input_shape=(None, 224, 224, 3))

    print("Starting Training...")
    model.train(images, labels)
    print("Finished Training!")

    model.save("checkpoints/weights_untrained.keras")

    predictions = model.predict(images)
    preds = np.argmax(predictions, axis=1)

    print(sum((preds[:1335]==0))/1335)
    print(sum((preds[1335:2670]==1))/1335)
    print(sum((preds[2670:4005]==2))/1335)
    print(sum((preds[4005:5340]==3))/1335)
    print(sum((preds[5340:6675]==4))/1335)

if __name__ == "__main__":
    main()
