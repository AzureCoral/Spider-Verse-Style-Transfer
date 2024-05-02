from preprocess import get_data
from models import TransferCNN
import tensorflow as tf

CHECKPOINT_PATH = "./checkpoints"

def main():
    all_data_folder = "transfer-learning/data"
    print("Starting Preprocessing...")
    images, labels = get_data(all_data_folder)
    print("Finished Preprocessing!")


    model = TransferCNN()

    print("Starting Training...")
    model.train(images, labels)
    print("Finished Training!")

    model.save_weights(CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
