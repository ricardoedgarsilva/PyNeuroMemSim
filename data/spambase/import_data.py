import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  

def import_data(test_size:float):  
    # fetch dataset 
    data = fetch_ucirepo(id=94)

    # data (as pandas dataframes) 
    x = data.data.features 
    y = data.data.targets

    # Manually creating one-hot encoded columns
    y['Class_0'] = (y['Class'] == 0).astype(int)  # Returns 1 for Class 0, otherwise 0
    y['Class_1'] = (y['Class'] == 1).astype(int)  # Returns 1 for Class 1, otherwise 0

    # Drop the original 'Class' column
    y.drop('Class', axis=1, inplace=True)

    # Normalize x to [0, 1] considering negative values
    x = (x - x.min()) / (x.max() - x.min())
  
    # merge x and y into data array in order to split into train and test sets
    data = np.concatenate((x, y), axis=1)
    train, test = train_test_split(data, test_size=test_size)

    # Split into x and y 
    x_train, y_train = train[:, :-2], train[:, -2:]
    x_test, y_test = test[:, :-2], test[:, -2:]

    return x_train, y_train, x_test, y_test


def post_processing(y_train, y_test):
    # List with probabilities, create a one hot encoding list with the highest probability of each row

    def one_hot_encode(y):
        one_hot_encoded = []
        for array in y:
            # Create an array of zeros with the same length as the input array
            one_hot = np.zeros_like(array)
            # Get the index of the maximum value in the original array
            index_of_max = np.argmax(array)
            # Set the corresponding index in the one-hot array to 1
            one_hot[index_of_max] = 1
            # Append the one-hot array to the list
            one_hot_encoded.append(one_hot)
        return one_hot_encoded
    
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return y_train, y_test




if __name__ == "__main__":
    x_train, y_train, x_test, y_test = import_data(0.2)

    print("Training data shape:", np.shape(x_train))
    print("Training labels shape:", np.shape(y_train))
    print("Test data shape:", np.shape(x_test))
    print("Test labels shape:", np.shape(y_test))




