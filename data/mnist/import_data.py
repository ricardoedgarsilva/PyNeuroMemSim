import numpy as np
from tensorflow.keras.datasets import mnist


if __name__ == "__main__":  
    

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Checking the shape of the datasets
    print("Training data shape:", train_images.shape)  # Output: (60000, 28, 28)
    print("Training labels shape:", train_labels.shape) # Output: (60000,)
    print("Test data shape:", test_images.shape)        # Output: (10000, 28, 28)
    print("Test labels shape:", test_labels.shape)      # Output: (10000,)



