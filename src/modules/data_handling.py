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
    Create and save input files for training and testing using multithreading.

    This function combines training and test data, then creates and saves these input values 
    as CSV files in a specified directory. Each file corresponds to a row of the input data.
    Multithreading is used to parallelize the creation of CSV files, with the number of threads 
    set to the number of CPU cores minus one, unless there is only one core.

    Parameters:
    x_train (np.array): The training data.
    x_test (np.array): The test data.
    config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    None
    """
    savedir = config["simulation"]["savedir"]

    try:
        print("Creating input files...")

        inputdir = os.path.join(savedir, "inputs")
        os.makedirs(inputdir, exist_ok=True)
        x = np.concatenate((x_test, x_train), axis=0)
        time = [config['simulation']['timestep'] * i for i in range(len(x))]
        
        def save_row(start_row, end_row):
            for row in range(start_row, end_row):
                inputs_train = pd.DataFrame(columns=['time', f'IN{row}'])
                inputs_train["time"] = time
                inputs_train[f'IN{row}'] = [x[i][row] for i in range(len(x))]
                inputs_train.to_csv(os.path.join(inputdir, f"in{row}.csv"), index=False, header=False)

        num_cpus = os.cpu_count() or 1  # Default to 1 if os.cpu_count() returns None
        num_threads = max(1, num_cpus - 1)  # Use one less than the number of CPU cores, unless there is only one core
        num_rows = config["simulation"]["geometry"][0][0]
        rows_per_thread = num_rows // num_threads
        threads = []

        for i in range(num_threads):
            start_row = i * rows_per_thread
            end_row = (i + 1) * rows_per_thread if i != num_threads - 1 else num_rows
            thread = threading.Thread(target=save_row, args=(start_row, end_row))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print("Input files created successfully!")

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

def backpropagate(config: dict, output_data, target_data):

    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]

    def sigmoid_derivative(x): 
        return x * (1 - x) * config["opamp"]["power"]

    # Calculate the error and deltas
    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1])

    new_weights = [np.zeros_like(w) for w in weights]

    for layer in reversed(range(len(weights))): 
        
        gradients = output_data[layer].T.dot(delta)

        new_weights[layer] = weights[layer] - learning_rate * gradients

        delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer])



    # Assuming there's a function to bound weights, call it 
    new_weights = bound_weights(new_weights)

    return new_weights

def rprop_update(config: dict, output_data, target_data):
    weights = config["simulation"]["weights"]
    
    # Constants for RPROP
    eta_plus = 1.2
    eta_minus = 0.5
    delta_max = 50
    delta_min = 0.000001
    
    # Initialize update values if not already present
    if "delta_values" not in config["simulation"]:
        config["simulation"]["delta_values"] = [0.1 * np.ones_like(w) for w in weights]

    deltas = config["simulation"]["delta_values"]
    last_gradients = config["simulation"].get("last_gradients", [np.zeros_like(w) for w in weights])

    def sigmoid_derivative(x): 
        return x * (1 - x) * config["opamp"]["power"]
    
    # Calculate the error and deltas
    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1])

    new_weights = [np.zeros_like(w) for w in weights]

    for layer in reversed(range(len(weights))):
        gradients = output_data[layer].T.dot(delta)

        # Update the step sizes based on the change in sign of the gradient
        sign_change = np.sign(gradients) * np.sign(last_gradients[layer])
        increase_indices = sign_change > 0
        decrease_indices = sign_change < 0
        deltas[layer][increase_indices] = np.minimum(deltas[layer][increase_indices] * eta_plus, delta_max)
        deltas[layer][decrease_indices] = np.maximum(deltas[layer][decrease_indices] * eta_minus, delta_min)
        
        # Update weights where gradient sign does not change
        weight_update_indices = sign_change >= 0
        new_weights[layer][weight_update_indices] = weights[layer][weight_update_indices] - np.sign(gradients[weight_update_indices]) * deltas[layer][weight_update_indices]
        
        # If the gradient sign changes, revert the weight updates
        new_weights[layer][decrease_indices] = weights[layer][decrease_indices]
        
        # Update deltas for next iteration
        last_gradients[layer] = gradients

        # Propagate delta to previous layer
        if layer > 0:
            delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer])

    # Store updated gradients and deltas back in config
    config["simulation"]["delta_values"] = deltas
    config["simulation"]["last_gradients"] = last_gradients

    # Optionally, bound weights if necessary
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


