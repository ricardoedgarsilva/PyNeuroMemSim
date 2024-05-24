from modules.dependencies import *

def import_data(config: dict):
    """
    Import training and testing data for a simulation.

    This function dynamically imports a module for data importation based on a configuration file. 
    It appends the specified data directory to the system path, imports the data import module, 
    and retrieves training and testing data based on the configuration settings.

    Parameters:
    config (dict): A dictionary containing configuration details, including the dataset path, 
                   test size, and geometry for the simulation.

    Returns:
    tuple: A tuple containing four elements - x_train, y_train, x_test, y_test.
    """
    
    try:
        print("\rImporting data...", end=' ' * 20)

        data_path = os.path.join("..", "data", config["simulation"]["dataset"])
        sys.path.append(data_path)

        module = importlib.import_module("import_data")
        impdt = getattr(module, "import_data")

        x_train, y_train, x_test, y_test = impdt(config["simulation"]["test_size"])

        # Add data shape to the configuration
        config["data_shape"] = {}
        config["data_shape"]["x_train"] = np.shape(x_train)
        config["data_shape"]["y_train"] = np.shape(y_train)
        config["data_shape"]["x_test"] = np.shape(x_test)
        config["data_shape"]["y_test"] = np.shape(y_test)

        # Performs check on data shapes and geometry compatibility

        def check_proceed():
            """
            Asks the user if they wish to proceed. If the user's response is not 'yes',
            the program will quit. If the user's response is 'yes', the program will proceed.
        
            Returns:
                None
            """
            # Prompt the user for input
            response = input("Do you wish to proceed? Type 'yes' to do so:\n")
        
            # Check the user's response
            if response.lower() != 'yes':
                print("\nQuitting...\n")
                quit()
            else:
                print("\nProceeding...\n")

        idtshp = np.shape(np.concatenate((x_test, x_train)))
        odtshp = np.shape(np.concatenate((y_test, y_train)))
        if idtshp[1] != config["simulation"]["geometry"][0][0]:
            print(f"ERROR: Input data shape {idtshp}  and input geometry ({config["simulation"]["geometry"][0][0]}) are incompatible.\n")
            check_proceed()
        elif odtshp[1] != config["simulation"]["geometry"][-1][1]:
            print(f"ERROR: Output data shape {odtshp} and output geometry ({config["simulation"]["geometry"][-1][1]}) are incompatible.\n")
            check_proceed()
        else:
            pass


        print("\rData imported successfully!", end=' ' * 20)

        return x_train, y_train, x_test, y_test
    except ImportError as ie:
        print(f"Error importing data module: {ie}")
    except AttributeError as ae:
        print(f"Error accessing 'import_data' function in the module: {ae}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None, None, None, None

def create_inputs(x_train, x_test, config):
    """
    Create and save input files for training and testing.

    This function combines training and test data, then creates and saves these input values 
    as CSV files in a specified directory. Each file corresponds to a row of the input data.

    Parameters:
    x_train (np.array): The training data.
    x_test (np.array): The test data.
    config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    None
    """

    savedir = config["simulation"]["savedir"]

    try:
        print("\rCreating input files...", end=' ' * 20)

        inputdir = os.path.join(savedir, "inputs")
        os.makedirs(inputdir, exist_ok=True)
        x = np.concatenate((x_test, x_train), axis=0)

        time = [config['simulation']['timestep'] * i for i in range(len(x))]

        for row in range(config["simulation"]["geometry"][0][0]):
            inputs_train = pd.DataFrame(columns=['time', f'IN{row}'])
            inputs_train["time"] = time
            inputs_train[f'IN{row}'] = [x[i][row] for i in range(len(x))]
            inputs_train.to_csv(os.path.join(inputdir, f"in{row}.csv"), index=False, header=False)
        
        print("\rInput files created successfully!", end=' ' * 20)

    except Exception as e:
        print(f"An error occurred while creating input files: {e}")

def calculate_mse(actual, predicted):
    """
    Calculate the Mean Squared Error between two arrays.
    
    Parameters:
    - actual (list or numpy array): The actual values.
    - predicted (list or numpy array): The predicted values.
    
    Returns:
    - float: The mean squared error.
    """

    print("\rCalculating Mean Squared Error...", end=' ' * 20)

    if len(actual) != len(predicted):
        raise ValueError(f"Both arrays must have the same length! {len(actual)} and {len(predicted)}")
    
    error = 0
    for a, p in zip(actual, predicted):
        error += (a - p) ** 2
    mse_value = error / len(actual)

    print("\rMean Squared Error calculated successfully!", end=' ' * 20)

    return np.mean(mse_value).round(5)

def calculate_time(len_xtrain, len_xtest, config):
    """
    Calculate the total simulation time based on the lengths of training and testing datasets and the simulation timestep.

    Parameters:
    len_xtrain (int): The length of the training dataset.
    len_xtest (int): The length of the testing dataset.
    config (dict): A dictionary containing the simulation configuration, specifically the timestep value.

    Returns:
    float: The total time required for the simulation.
    """

    timestep = config["simulation"]["timestep"]
    
    return (len_xtrain + len_xtest) * timestep

def split_data(data, val_len):
    """
    Splits each sequence in the input data into two parts: validation and training.

    The function iterates over a list of sequences (`data`), and for each sequence,
    it extracts two parts: the first `val_len` elements for the validation set, and the
    remaining elements for the training set.

    Args:
    data (list of lists/tuples): A list where each element is a sequence from which
                                 the first `val_len` elements are taken as validation data.
    val_len (int): The number of elements from the start of each sequence to include in
                   the validation data.

    Returns:
    tuple of two lists: (validation_data, training_data)
        - validation_data: A list containing the `val_len` initial elements of each sequence.
        - training_data: A list containing the remaining elements of each sequence after the
                         initial `val_len` elements.
    """

    print("\rSplitting imported data into validation/training ...", end=' ' * 20)

    # Use list comprehension for concise and efficient data splitting
    validation_data = [sequence[:val_len] for sequence in data]
    training_data = [sequence[val_len:] for sequence in data]

    print("\rData split successfully!", end=' ' * 20)

    return validation_data, training_data

def create_mse_hist(config: dict):
    """
    Initializes a CSV file to store the MSE values across different epochs.

    This function takes a configuration dictionary that specifies the directory where
    the MSE history CSV file should be saved. It then creates or overwrites an existing
    file named 'mse_hist.csv' in the specified directory with a header for logging
    MSE values for validation and training during different epochs.

    Args:
    config (dict): Configuration dictionary with nested dictionary 'simulation' containing:
                   - 'savedir': A string that specifies the directory path where the CSV
                                file will be saved.

    Returns:
    None
    """

    print("\rCreating MSE history CSV file...", end=' ' * 20)

    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "mse_hist.csv")

    with open(csv_path, "w") as f:
        # Write the header of the CSV file
        f.write("epoch,mse_val,mse_trn\n")
    
    print("\rMSE history CSV file created successfully!", end=' ' * 20)

def append_mse_hist(config: dict, epoch: int, mse_val: float, mse_trn: float):
    """
    Appends the MSE values for a specific epoch to an existing CSV file.

    This function uses a configuration dictionary to determine the save directory for the CSV file.
    It appends a new line with the current epoch and corresponding MSE values for validation and 
    training to 'mse_hist.csv'.

    Args:
    config (dict): Configuration dictionary with a nested dictionary 'simulation' that includes:
                   - 'savedir': A string specifying the directory path where the CSV file is located.
    epoch (int): The current epoch number.
    mse_val (float): The MSE value for the validation dataset in the current epoch.
    mse_trn (float): The MSE value for the training dataset in the current epoch.

    Returns:
    None
    """

    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "mse_hist.csv")

    with open(csv_path, "a") as f:
        # Write the new epoch data to the CSV file
        f.write(f"{epoch},{mse_val},{mse_trn}\n")


#-------- Experimenting several backpropagation algorithms
def bound_weights(weights: list):
    # Bound weights between 0 and 1
    for i in range(len(weights)):
        weights[i] = np.clip(weights[i], 0.01, 0.99)
    return weights


def backpropagate_old(config, trn_data, trn_out, weights, learning_rate):

    def sigmoid_derivative(x): return x * (1 - x) * config["opamp"]["power"]

    layer_errors = [trn_out - trn_data[-1]]
    layer_deltas = [layer_errors[0] * sigmoid_derivative(trn_data[-1])]

    for i in range(len(weights) - 1, -1, -1):
        error = layer_deltas[-1].dot(weights[i].T)
        delta = error * sigmoid_derivative(trn_data[i])
        layer_errors.append(error)
        layer_deltas.append(delta)

    # Reverse the error and delta lists
    layer_errors.reverse()
    layer_deltas.reverse()

    layer_deltas.pop(0)
    layer_errors.pop(0)

    # Update weights
    for i in range(len(weights)): 

        dif =  trn_data[i].T.dot(layer_deltas[i])
        
        if np.mean(layer_deltas[i]) <= 0: sign = 1
        else: sign = -1

        weights[i] += learning_rate * sign * dif

    return weights

def backpropagate(config: dict, output_data, target_data):

    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]

    def sigmoid_derivative(x): return x * (1 - x) * config["opamp"]["power"]

    # Initialize the list to store the gradient of weights for each layer
    gradients = [np.zeros_like(w) for w in weights]

    # Start with the output layer
    error = output_data[-1] - target_data
    delta = error * sigmoid_derivative(output_data[-1])	

    # Loop in reverse order to backpropagate the error
    for layer in reversed(range(len(weights))):

        
        # Calculate the gradient for the current layer
        gradients[layer] = output_data[layer].T.dot(delta)

        # Propagate the error backwards
        if layer > 0:
            delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer])
        
    # Update the weights using the calculated gradients
    new_weights = [w + learning_rate * grad for w, grad in zip(weights, gradients)]

    # Bound the weights between 0 and 1
    # new_weights = bound_weights(new_weights)

    return new_weights

def backpropagate2(config: dict, output_data, target_data):
    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]

    def sigmoid_derivative(x): 
        return x * (1 - x) * config["opamp"]["power"]

    # Initialize the list to store the gradient of weights for each layer
    gradients = [np.zeros_like(w) for w in weights]

    # Start with the output layer
    error = output_data[-1] - target_data
    delta = error * sigmoid_derivative(output_data[-1])

    # Loop in reverse order to backpropagate the error
    for layer in reversed(range(len(weights))):
        # Calculate the gradient for the current layer using element-wise multiplication for delta and output
        gradients[layer] = np.multiply.outer(delta, output_data[layer])

        # Propagate the error backwards
        if layer > 0:
            # Calculate delta for next layer; re-adjust the shape for broadcasting, if necessary
            delta = np.sum(delta[:, None, :] * weights[layer][None, :, :], axis=2) * sigmoid_derivative(output_data[layer])

    # Update the weights using the calculated gradients element-wise
    new_weights = [w - learning_rate * grad for w, grad in zip(weights, gradients)]

    return new_weights

def backpropagate3(config, output_data, target_data):
    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]

    def sigmoid_derivative(x): return x * (1 - x) * config["opamp"]["power"]

    # Calculate initial output layer delta
    # Assuming sigmoid activation function
    error = (output_data[-1] - target_data)
    deltas = [error * sigmoid_derivative(output_data[-1])]
    
    # Backpropagate the errors
    for i in reversed(range(len(weights) - 1)):
        current_output = output_data[i + 1]
        derivative = current_output * (1 - current_output)
        error = deltas[0]
        propagated_error = np.dot(error, weights[i + 1].T) * derivative
        deltas.insert(0, propagated_error)
    
    # Update weights layer by layer
    new_weights = []
    for i in range(len(weights)):
        weight = weights[i]
        updated_weight = np.copy(weight)
        layer_input = output_data[i]
        delta = deltas[i]
        
        # Update each weight individually
        for r in range(weight.shape[0]):
            for c in range(weight.shape[1]):
                # Calculate gradient as mean of product of inputs and delta errors
                gradient = np.mean(layer_input[:, r] * delta[:, c])
                # Ensure gradient is a scalar
                gradient = gradient.item() 
                updated_weight[r, c] += learning_rate * gradient
        
        new_weights.append(updated_weight)

    # Bound the weights between 0 and 1
    new_weights = bound_weights(new_weights)
    
    return new_weights

def backpropagate4(config, output_data, target_data):
    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]

    def sigmoid_derivative(x): 
        return x * (1 - x) * config["opamp"]["power"]

    def sign(x):
        return np.sign(x)

    # Calculate initial output layer delta
    # Assuming sigmoid activation function
    error = (output_data[-1] - target_data)
    deltas = [error * sigmoid_derivative(output_data[-1])]
    
    # Backpropagate the errors
    for i in reversed(range(len(weights) - 1)):
        current_output = output_data[i + 1]
        derivative = sigmoid_derivative(current_output)
        error = deltas[0]
        propagated_error = np.dot(error, weights[i + 1].T) * sign(derivative)
        deltas.insert(0, propagated_error)
    
    # Update weights layer by layer
    new_weights = []
    for i in range(len(weights)):
        weight = weights[i]
        updated_weight = np.copy(weight)
        layer_input = output_data[i]
        delta = deltas[i]
        
        # Update each weight individually
        for r in range(weight.shape[0]):
            for c in range(weight.shape[1]):
                # Calculate gradient as mean of product of inputs and delta errors
                gradient = np.mean(layer_input[:, r] * delta[:, c])
                # Ensure gradient is a scalar
                gradient = gradient.item()
                updated_weight[r, c] += learning_rate * gradient
        
        new_weights.append(updated_weight)

    # Bound the weights between 0 and 1
    new_weights = bound_weights(new_weights)
    
    return new_weights

#-------- Needs implementation

def create_weight_hist(config: dict, weights: list):
    path = os.path.join(config["simulation"]["savedir"], "weight_hist")
    os.makedirs(path)

    with h5py.File(os.path.join(path, "weights.hdf5"), "a") as f:
        for layer_idx, weight_array in enumerate(weights):

            # Create dataset 
            f.create_dataset(
                f'layer_{layer_idx}', 
                data = weight_array, 
                compression = "gzip",
                dtype = 'float32'
            )

            dataset = f[f'layer_{layer_idx}']
            dataset[0] = weight_array

def append_weight_hist(config: dict, weights: list):

    with h5py.File(os.path.join(config["simulation"]["savedir"], "weight_hist", "weights.hdf5"), "a") as f:
        for layer_idx, weight_array in enumerate(weights):
            dataset = f[f'layer_{layer_idx}']
            

            current_size = dataset.shape[0]
            new_size = current_size + 1
            dataset.resize(new_size, axis=0)

            dataset[current_size] = weight_array


