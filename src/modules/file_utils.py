from modules.dependencies import *

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
    from modules.info import info

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
