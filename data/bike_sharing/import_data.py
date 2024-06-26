import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  

def import_data(test_size:float):  
    
    # fetch dataset 
    data = fetch_ucirepo(id=275)

    # data (as pandas dataframes) 
    x = data.data.features 
    y = data.data.targets
    y_cols = len(y.columns)

    # Remove second column (dteday) from x
    x = x.drop(columns=['dteday'])


    # Normalize x and y to [0, 1] considering negative values
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # merge x and y into data array in order to split into train and test sets
    data = np.concatenate((x, y), axis=1)
    train, test = train_test_split(data, test_size=test_size)

    # Split into x and y
    x_train, y_train = train[:, :-y_cols], train[:, -y_cols:]
    x_test, y_test = test[:, :-y_cols], test[:, -y_cols:]

    # Round y_train and y_test to 2 decimal places
    y_train = np.round(y_train,4)
    y_test = np.round(y_test,4)


    return x_train, y_train, x_test, y_test


def post_processing(y_train, y_test):
    # List with probabilities, create a one hot encoding list with the highest probability of each row
    
    y_train = np.round(y_train,4)
    y_test = np.round(y_test,4)

    return y_train, y_test



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = import_data(0.2)

    print("Training data shape:", np.shape(x_train))
    print("Training labels shape:", np.shape(y_train))
    print("Test data shape:", np.shape(x_test))
    print("Test labels shape:", np.shape(y_test))






