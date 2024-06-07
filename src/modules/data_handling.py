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

def scale_data(config: dict, x_train, y_train, x_test, y_test):
    """
    Scale the input and output data based on the configuration settings.

    This function scales the input and output data using the specified scaling factors in the configuration dictionary.
    The scaling factors are applied to the training and testing data, and the scaled data is returned.

    Parameters:
    config (dict): Configuration dictionary containing the scaling factors.
    x_train (np.array): The training input data.
    y_train (np.array): The training output data.
    x_test (np.array): The test input data.
    y_test (np.array): The test output data.

    Returns:
    tuple: A tuple containing four elements - x_train, y_train, x_test, y_test.
    """

    print("\rScaling data...", end=' ' * 20)

    [x_scale, y_scale] = config["simulation"]["xy_scale"]
    x_train = x_train * x_scale
    y_train = y_train * y_scale
    x_test = x_test * x_scale
    y_test = y_test * y_scale

    print("\rData scaled successfully!", end=' ' * 20)

    return x_train, y_train, x_test, y_test

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

def apply_postprocessing(config: dict, trn_data, val_data):
    """
    Applies a post-processing function to training and validation data. The function is dynamically
    imported from a module specified in the given configuration dictionary.

    Parameters:
    - config (dict): Configuration dictionary containing details such as the dataset and the
                     module name for post-processing.
    - trn_data: Training data on which the post-processing function will be applied.
    - val_data: Validation data on which the post-processing function will be applied.

    Returns:
    - tuple: A tuple containing the post-processed training and validation data. If the
             post-processing function cannot be imported, it returns the original training
             and validation data.

    Raises:
    - ImportError: If the specified post-processing module or function is not found.

    The function first constructs the path to the dataset based on the configuration dictionary.
    It then attempts to import a module named 'import_data' from this path and retrieves a function
    named 'post_processing' from the imported module. This function is then called with the
    training and validation data. If successful, it prints a success message and returns the
    processed data. In case of an ImportError, it prints an error message and returns the
    original data.
    """

    data_path = os.path.join("..", "data", config["simulation"]["dataset"])
    sys.path.append(data_path)

    try:
        module = importlib.import_module("import_data")
        impdt = getattr(module, "post_processing")

        y_train, y_test = impdt(trn_data, val_data)

        print("\rPost-processing applied successfully!", end=' ' * 20)

        return y_train, y_test
    except:
        print(f"\rNo post-processing function found in the module 'import_data'.", end=' ' * 20)
        
        return trn_data, val_data
        
def bound_weights(config: dict):
    """
    Bound each weight in a list to be within a specific range using NumPy's clip function.
    
    This ensures that each weight is constrained to be no less than 0.01 and no more than 0.99,
    which is typically done to avoid weights that are exactly 0 or 1 in certain machine learning algorithms.
    
    Parameters:
    - config (dict): Configuration dictionary containing the weights to be bounded.
    
    Returns:
    - None: The weights in the configuration dictionary are updated in-place.
    """
    # Use numpy's clip function to bound each weight within the specified range.
    [lbond, ubond] = config["learning"]["bound_limits"]
    config["simulation"]["weights"] = [np.clip(weight, lbond, ubond) for weight in config["simulation"]["weights"]]

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

    match config["memristor"]["xo_relation"]:
        case "direct":
            # case for x
            pass
        case "indirect":
            # case for 1-x
            weights = [1 - layer for layer in weights]
        case "inverse":
            # case for 1/x
            weights = [1 / layer for layer in weights]
        case _:
            raise ValueError("Invalid weight-xo relation specified in the configuration.")


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
