import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  

def import_data(test_size:float):  
    # fetch dataset 
    data = fetch_ucirepo(id=697)

    # data (as pandas dataframes) 
    x = data.data.features 
    y = data.data.targets

    # Convert y to one-hot encoding
    y = pd.get_dummies(y).astype(int)

    # Normalize x to [0, 1] considering negative values
    x = (x - x.min()) / (x.max() - x.min())
  
    # merge x and y into data array in order to split into train and test sets
    data = np.concatenate((x, y), axis=1)
    train, test = train_test_split(data, test_size=test_size)

    # Split into x and y 
    x_train, y_train = train[:, :-3], train[:, -3:]
    x_test, y_test = test[:, :-3], test[:, -3:]


    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = import_data(0.2)

    print("Training data shape:", np.shape(x_train))
    print("Training labels shape:", np.shape(y_train))
    print("Test data shape:", np.shape(x_test))
    print("Test labels shape:", np.shape(y_test))

    # Check for negative values



