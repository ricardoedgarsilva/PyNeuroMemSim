from modules.dependencies import *
from modules.data_handling import apply_postprocessing

def create_mtr_hist(config: dict):
    """
    Initializes a CSV file to store the metric values across different epochs.

    This function takes a configuration dictionary that specifies the directory where
    the MSE history CSV file should be saved. It then creates or overwrites an existing
    file named 'metrics.csv' in the specified directory with a header for logging
    MSE values for validation and training during different epochs.

    Args:
    config (dict): Configuration dictionary with nested dictionary 'simulation' containing:
                   - 'savedir': A string that specifies the directory path where the CSV
                                file will be saved.

    Returns:
    None
    """

    print("\rCreating metric history CSV file...", end=' ' * 20)

    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "metrics.csv")

    header = ["epoch"]

    for metric in config["learning"]["metrics"]:
        header.append(f"{metric}_val")
        header.append(f"{metric}_trn")

    with open(csv_path, "w") as f:
        # Write the header of the CSV file
        f.write(",".join(header) + "\n")
    
    print("\rMetric history CSV file created successfully!", end=' ' * 20)

def append_mtr_hist(config: dict, metrics):
    """
    Appends the metric values for a specific epoch to an existing CSV file.

    This function uses a configuration dictionary to determine the save directory for the CSV file.
    It appends a new line with the current epoch and corresponding metric values for validation and 
    training to 'metrics.csv'.

    Args:
    config (dict): Configuration dictionary with a nested dictionary 'simulation' that includes:
                   - 'savedir': A string specifying the directory path where the CSV file is located.
    metrics (list): A list of metric values for the current epoch. The first element is the epoch number.

    Returns:
    None
    """

    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "metrics.csv")

    with open(csv_path, "a") as f:
        # Write the new epoch data to the CSV file
        f.write(",".join(map(str, metrics)) + "\n")

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

def calculate_f1(actual, predicted):
    """
    Calculate the F1 score between two arrays of actual and predicted binary classifications.

    Parameters:
    - actual (list or numpy array): The actual binary classification values.
    - predicted (list or numpy array): The predicted binary classification values.

    Returns:
    - float: The F1 score rounded to five decimal places.

    Raises:
    - ValueError: If the input arrays do not have the same length.

    The F1 score is calculated as 2 * (precision * recall) / (precision + recall),
    where precision is the number of true positives divided by the number of all positive predictions,
    and recall is the number of true positives divided by the number of all actual positives.
    """
    print("\rCalculating F1 Score...", end=' ' * 20)

    # Convert inputs to numpy arrays and flatten them
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    if len(actual) != len(predicted):
        raise ValueError(f"Both arrays must have the same length! {len(actual)} and {len(predicted)}")

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            true_positive += 1
        elif a == 0 and p == 1:
            false_positive += 1
        elif a == 1 and p == 0:
            false_negative += 1

    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print("\rF1 Score calculated successfully!", end=' ' * 20)

    return round(f1_score, 5)

def calculate_metrics(config:dict, trn_data, val_data, y_train, y_test):
    """
    Calculate the metrics specified in the configuration dictionary.

    This function calculates the metrics specified in the 'metrics' key of the 'learning' dictionary
    in the configuration dictionary. The function uses the training and validation data and the actual
    target values to calculate the specified metrics. The function returns a dictionary with the metric
    names as keys and the calculated metric values as values.

    Args:
    config (dict): Configuration dictionary with a nested dictionary 'learning' that includes:
                   - 'metrics': A list of metric names to calculate.
    trn_data (numpy array): The predicted output values for the training data.
    val_data (numpy array): The predicted output values for the validation data.
    y_train (numpy array): The actual target values for the training data.
    y_test (numpy array): The actual target values for the validation data.

    Returns:
    dict: A dictionary with the metric names as keys and the calculated metric values as values.
    """

    metrics = [config["simulation"]["epoch"]]

    for metric in config["learning"]["metrics"]:
        if metric == "mse":
            metrics.append(calculate_mse(y_test, val_data))
            metrics.append(calculate_mse(y_train, trn_data))

        elif metric == "f1_score":
            y_trnpp, y_valpp = apply_postprocessing(config, trn_data, val_data)
            metrics.append(calculate_f1(y_test, y_valpp))
            metrics.append(calculate_f1(y_train, y_trnpp))

        else:
            raise ValueError(f"Metric {metric} not recognized!")

    return metrics

def print_metric_info(config:dict, metrics:list):
    """
    Print the metric values for the current epoch.

    This function prints the metric values for the current epoch to the console. The function
    uses the configuration dictionary to determine the metric names and the epoch number. The
    function prints the epoch number and the metric values for validation and training data.

    Args:
    config (dict): Configuration dictionary with a nested dictionary 'learning' that includes:
                 - 'metrics': A list of metric names to calculate.
    metrics (list): A list of metric values for the current epoch.

    Returns:
    None
    """

    # Print information about the epoch and the metrics
    epoch = metrics[0]
    times = config["simulation"]["times"]
    str_metrics = [f"\rEpoch {epoch:3}"]

    for metric in range(len(config["learning"]["metrics"])):
        str_metrics.append(f"{config['learning']['metrics'][metric]} val: {metrics[2*metric+1]:0.5f}")
        str_metrics.append(f"{config['learning']['metrics'][metric]} trn: {metrics[2*metric+2]:0.5f}")
    
    str_metrics.append(f"Time: {times[0]} s")
    str_metrics.append(f"ET: {np.round(np.mean(times) * (config["simulation"]["epochs"] - epoch), 2)	} s")
    
    print(" | ".join(str_metrics))