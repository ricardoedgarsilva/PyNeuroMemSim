import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  

def import_data(test_size:float):  
    # fetch dataset 
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes) 
    x = wine_quality.data.features 
    y = wine_quality.data.targets


    x = x / np.max(x)
    y = y / np.max(y)    
  
    # merge x and y into data array in order to split into train and test sets
    data = np.concatenate((x, y), axis=1)
    train, test = train_test_split(data, test_size=test_size)
    

    # Split into x and y again considering that len(y) = 1
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    # Convert y's to one-hot encoding up to 10 classes
    y_train = np.eye(10)[y_train.astype(int)]
    y_test = np.eye(10)[y_test.astype(int)]


    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    import_data(0.2, None)


