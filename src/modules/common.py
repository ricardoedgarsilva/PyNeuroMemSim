import os
import sys
import h5py
import datetime
import importlib
import subprocess
import ltspice

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def ensure_directory_is_src():
    """
    Ensures that the current working directory is set to a subdirectory named 'src'.
    If the current working directory is not 'src', it changes the directory to 'src'
    nested inside the current working directory.

    Raises:
        FileNotFoundError: If the 'src' directory does not exist within the current working directory.
        PermissionError: If the program does not have permission to change the working directory.
    """
    # Get the current working directory
    cwd = os.getcwd()
    
    # Check if the last component of the current directory is 'src'
    if os.path.basename(cwd) != "src":
        src_path = os.path.join(cwd, "src")
        # Attempt to change the working directory to 'src'
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"The directory '{src_path}' does not exist.")
        os.chdir(src_path)
        print(f"Changed working directory to {src_path}")
    else:
        print("Current working directory is already 'src'.")

def printlog_info(config: dict):
    """
    Writes a detailed log of the simulation configuration to a file and prints it.

    This function generates a report containing the simulator's metadata and the detailed
    configuration settings, excluding specified keys. The log is saved to a file named
    'config.log' in the specified directory and also printed to the console.

    Args:
    config (dict): Configuration dictionary containing all the settings of the simulation,
                   nested within 'simulation'.

    Returns:
    None
    """

    # Define the directory to save the log file
    savedir = config["simulation"]["savedir"]
    # Keys to exclude from logging
    key_exclusions = ['weights', 'subcircuits']

    # Construct the header information for the log
    from modules.common import info

    # Function to log information to both file and console
    def printlog(f, prt):
        f.write(f"{prt}\n")
        print(prt)

    # Function to recursively print dictionary content, excluding certain keys
    def print_dict_with_titles(f, d):
        for key, value in d.items():
            if key in key_exclusions:
                continue
            elif isinstance(value, dict):
                printlog(f, f"\n---- {key} ----")
                print_dict_with_titles(f, value)  # Recursive call for nested dictionaries
            else:
                printlog(f, f"{key}: {value}")

    # Open the log file in write mode and start logging
    with open(os.path.join(savedir, "config.log"), "w") as f:
        printlog(f, info)
        print_dict_with_titles(f, config)

    # Print a closing line to the console
    print(
        f"\n {10 * '-'} \n",
        2 * "\n",
    )

def create_directory(directory: str):
    """
    Create a new directory at the specified path if it doesn't exist.

    This function creates a directory at the given path. If the directory already exists, no action is taken. 
    If any part of the path does not exist, it will attempt to create the entire directory structure.

    Parameters:
    directory (str): The path where the directory should be created.

    Returns:
    None
    """
    
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' created successfully.")
    except OSError as e:
        print(f"Creation of the directory '{directory}' failed. Reason: {e}")

def create_save_directory(directory: str):
    """
    Create a subdirectory with the specified name within the given directory and return its path.

    This function takes a directory path and a folder name, then creates a subdirectory with that name inside the given directory.
    If the specified directory or subdirectory does not exist, they are created. If they already exist, no additional action is taken.
    The function then returns the path of the created or existing subdirectory.

    Parameters:
    directory (str): The path of the directory where the subdirectory should be created.
    foldername (str): The name of the subdirectory to create.

    Returns:
    str: The path of the created or existing subdirectory.
    """
    
    try:

        # Create the main directory if it doesn't exist
        if not os.path.isdir(directory): create_directory(directory)

        # Count the amount of subdirectories in the path
        count = len([name for name in os.listdir(directory)])

        # Create a new folder with the current date and time
        foldername = f"SIM{count}_{datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}"
        
        subdirectory_path = os.path.join(directory, foldername)
        os.makedirs(subdirectory_path, exist_ok=True)
        
    except OSError as e:
        print(f"Creation of the subdirectory '{subdirectory_path}' failed. Reason: {e}")
        return None

    return subdirectory_path

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
        inputdir = os.path.join(savedir, "inputs")
        os.makedirs(inputdir, exist_ok=True)
        x = np.concatenate((x_test, x_train), axis=0)

        time = [config['simulation']['timestep'] * i for i in range(len(x))]

        for row in range(config["simulation"]["geometry"][0][0]):
            inputs_train = pd.DataFrame(columns=['time', f'IN{row}'])
            inputs_train["time"] = time
            inputs_train[f'IN{row}'] = [x[i][row] for i in range(len(x))]
            inputs_train.to_csv(os.path.join(inputdir, f"in{row}.csv"), index=False, header=False)

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

def initialize_weights(config: dict):
    """
    Initialize random weights for each node in a given geometry.

    This function creates a nested list of weights where each sublist corresponds to a layer in the geometry,
    and each sub-sublist corresponds to the nodes in that layer. The weights are initialized randomly.

    Parameters:
    config (dict): A dictionary containing the simulation configuration, specifically the geometry of the network.

    Returns:
    list: A nested list containing the weights for each node in each layer of the geometry.
    """
    geometry = config["simulation"]["geometry"]

    weights = [np.random.rand(rows, cols) for rows, cols in geometry]

    return weights

def run_ltspice(config: dict, mode="-b"):
    """
    Run an LTspice simulation using the specified circuit file.

    This function runs an LTspice simulation using the specified circuit file. 
    It opens an LTspice subprocess and runs the simulation with the given circuit file.

    Parameters:
    config (dict): A dictionary containing the configuration details for the simulation, including the path to the circuit file.

    Returns:
    None
    """

    circuit_path = os.path.join(config["simulation"]["savedir"], "circuit.cir")

    
    try:
        if mode != "-b" and mode != "":
            raise ValueError("Invalid mode. Use '-b' for background or '' for foreground.")

        subprocess.run(["ltspice", mode, circuit_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"An error occurred while running LTspice: {e}")

def import_results(config):
    """
    Import simulation results from an LTspice simulation.

    This function reads simulation data from a .raw file generated by LTspice, based on the 
    configuration specified in `config`. It extracts voltage data at specific intervals
    for different layers and columns in a simulation geometry.

    Parameters:
    config (dict): A dictionary containing configuration data with keys:
                   - 'simulation': a sub-dictionary containing:
                     - 'savedir': the directory where simulation results are saved.
                     - 'geometry': a list of tuples, where each tuple represents the
                       number of rows and columns for each layer in the simulation.
                     - 'freq': the frequency at which data points are sampled.

    Returns:
    list: A list of numpy arrays. Each array corresponds to a layer in the simulation
          geometry, containing voltage data for each column at specified intervals.
    """


    # Extract configuration details
    save_dir = config['simulation']['savedir']
    geometry = config['simulation']['geometry']
    precision = config['simulation']['precision']
    timesteps = config['data_shape']['x_train'][0] + config['data_shape']['x_test'][0]

    # Load and parse the LTspice data
    data_file_path = os.path.join(save_dir, "circuit.raw")
    data = ltspice.Ltspice(data_file_path)
    data.parse()

    # Process time data and find indices
    time_data = np.round(data.get_data("time"), precision)
    time_list = [config['simulation']['timestep'] * i for i in range(timesteps)]

    # Initialize results list and add the output layer
    results = [np.zeros((timesteps, rows)) for rows, _ in geometry]
    results.append(np.zeros((timesteps, geometry[-1][1])))

    # Extract voltage data for each layer row
    for layer in range(0, len(geometry)):
        for row in range(0, geometry[layer][0]):
            voltage_header = f"V(nin_{layer}_{row})"
            voltage_data = data.get_data(voltage_header)
            f = interp1d(time_data, voltage_data)
            results[layer][:,row] = f(time_list)
        
    # Extract voltage data of the system output
    layer = len(geometry)
    for col in range(geometry[-1][1]):
        voltage_header = f"V(nin_{layer}_{col})"
        voltage_data = data.get_data(voltage_header)
        f = interp1d(time_data, voltage_data)
        results[-1][:,col] = f(time_list)


    return results

def calculate_mse(actual, predicted):
    """
    Calculate the Mean Squared Error between two arrays.
    
    Parameters:
    - actual (list or numpy array): The actual values.
    - predicted (list or numpy array): The predicted values.
    
    Returns:
    - float: The mean squared error.
    """
    if len(actual) != len(predicted):
        raise ValueError(f"Both arrays must have the same length! {len(actual)} and {len(predicted)}")
    
    error = 0
    for a, p in zip(actual, predicted):
        error += (a - p) ** 2
    mse_value = error / len(actual)
    return np.mean(mse_value).round(3)

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

    # Use list comprehension for concise and efficient data splitting
    validation_data = [sequence[:val_len] for sequence in data]
    training_data = [sequence[val_len:] for sequence in data]

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

    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "mse_hist.csv")

    with open(csv_path, "w") as f:
        # Write the header of the CSV file
        f.write("epoch,mse_val,mse_trn\n")

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

def plot_mse_hist(config: dict):
    """
    Reads the MSE data from a CSV file and plots the MSE for validation and training across epochs.

    This function generates a line plot of the MSE values for validation and training from a CSV
    file named 'mse_hist.csv', located in a directory specified by the configuration dictionary.
    The plot is saved as an image file 'mse_hist.png' in the same directory.

    Args:
    config (dict): Configuration dictionary containing a nested dictionary 'simulation' which includes:
                   - 'savedir': A string specifying the directory path where the CSV file is located and
                                where the plot image will be saved.

    Returns:
    None

    Raises:
    FileNotFoundError: If the 'mse_hist.csv' file does not exist in the specified directory.
    Exception: For issues related to reading the CSV or plotting.
    """
    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "mse_hist.csv")

    try:
        # Load MSE data from the CSV file
        data = pd.read_csv(csv_path)

        # Create a line plot for MSE values
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size for better visibility
        plt.plot(data["epoch"], data["mse_val"], label="Validation")
        plt.plot(data["epoch"], data["mse_trn"], label="Training")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE per Epoch for Validation and Training")  # Optional: Add a title to the plot
        plt.legend()
        plt.grid(True)  # Optional: Add a grid for easier reading

        # Save the plot as an image file
        plt.savefig(os.path.join(savedir, "mse_hist.png"))

        # Close the plot to free up memory
        plt.close()

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while plotting MSE: {e}")


#-------- Needs documentation and improvement

def backpropagate_matrix(config, trn_data, trn_out, weights, learning_rate):

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
    new_weights = [w - learning_rate * grad for w, grad in zip(weights, gradients)]

    return new_weights



#-------- Needs implementation

def plot_weight_evolution(config: dict, weights: list):

    path = os.path.join(config["simulation"]["savedir"], "weight_evolution")
    os.makedirs(path)

    for layer_idx, layer in enumerate(weights):
        fig, ax = plt.subplots()
        cax = ax.matshow(layer, interpolation='nearest', cmap='bwr', vmin=0, vmax=1)
        fig.colorbar(cax)

        def update(epoch):
            cax.set_data(layer[epoch])
            ax.set_title(f'Epoch {epoch}')
            ax.set_aspect('equal', 'box')
            return cax,

        ani = FuncAnimation(fig, update, frames=config['simulation']['epochs'], repeat=True)
        ani.save(os.path.join(path, f"l{layer_idx}_weights.mp4"), writer='ffmpeg')

        plt.close(fig)

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


