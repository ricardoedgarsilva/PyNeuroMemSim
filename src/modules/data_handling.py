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

def bound_weights(weights):
    """
    Bound each weight in a list to be within a specific range using NumPy's clip function.
    
    This ensures that each weight is constrained to be no less than 0.01 and no more than 0.99,
    which is typically done to avoid weights that are exactly 0 or 1 in certain machine learning algorithms.
    
    Parameters:
    - weights (list of float): The list of weight values to be bounded.
    
    Returns:
    - list of float: The list of bounded weights.
    """
    # Use numpy's clip function to bound each weight within the specified range.
    # This modifies the list in-place and returns the modified list.
    return [np.clip(weight, 0.01, 0.99) for weight in weights]

def compute_weight_histogram(config: dict):
    """
    Compute weight histograms for the given simulation configuration.
    
    The function assumes a structure of `config` dictionary where weights and bin size
    are provided under the `simulation` key. It computes the histograms for the flattened
    weights of each layer and appends these histograms to the `weight_distribution` list
    in the config.

    Parameters:
    config (dict): Configuration dictionary with keys:
        - simulation: A dictionary containing:
            - weights (list of arrays): Weight matrices of different layers.
            - bin_size (float): Size of the bin for histogram computation.
            - weight_distribution (list): A list to store histograms of weight distributions.

    Prints:
    Status of the computation with updates.
    """

    print("\rComputing weight histograms", end=' ' * 20)
    
    if "weight_distribution" not in config["simulation"]:
        config["simulation"]["weight_distribution"] = []

    histograms = []
    bin_size = config["simulation"]["bin_size"]
    bins = np.arange(0, 1+bin_size, bin_size)
    weights = config["simulation"]["weights"]

    for layer in weights:

        hist, _ = np.histogram(layer.flatten(), bins=bins)
        histograms.append(hist)

    config["simulation"]["weight_distribution"].append(histograms)

    print("\rWeight histograms computed successfully!", end=' ' * 20)

def save_weight_histogram(config: dict):
    """
    Save the computed weight histograms to a file.

    Assumes the weight histograms are stored under `simulation -> weight_distribution` in the
    provided configuration dictionary. Saves the histograms as a pickle file in the specified
    directory.

    Parameters:
    config (dict): Configuration dictionary with keys:
        - simulation: A dictionary containing:
            - savedir (str): Directory path to save the histogram file.
            - weight_distribution (list): List of histograms to be saved.

    Prints:
    Status of the file save operation.
    """

    print("\rSaving weight histograms", end=' ' * 20)

    filepath = os.path.join(config["simulation"]["savedir"], "weight_distribution.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(config["simulation"]["weight_distribution"], f)

    print("\rWeight histograms saved successfully!", end=' ' * 20)
