import numpy as np
from keras.datasets import mnist

def import_data(test_size:float):

    print("For MNIST dataset, test_size is not used!")


    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape images to (len, 784)
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Normalize images to [0, 1] 
    train_images = train_images / 255
    test_images = test_images / 255

    # Convert labels to one-hot encoding
    train_labels = np.eye(10)[train_labels.astype(int)]
    test_labels = np.eye(10)[test_labels.astype(int)]

    return train_images, train_labels, test_images, test_labels



if __name__ == "__main__":  
    

    x_train, y_train, x_test, y_test = import_data(0.2)

    print("Training data shape:", np.shape(x_train))
    print("Training labels shape:", np.shape(y_train))
    print("Test data shape:", np.shape(x_test))
    print("Test labels shape:", np.shape(y_test))

    





