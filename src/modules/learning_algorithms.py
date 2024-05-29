from modules.dependencies import *

def sigmoid_derivative(x, config: dict): 
    """
    Calculate the derivative of the sigmoid function scaled by a power factor.
    
    The sigmoid function derivative is given by x * (1 - x). This function
    further scales the derivative by a power factor specified in the configuration.
    
    Parameters:
    - x (float): The input value where the derivative is to be calculated.
    - config (dict): Configuration dictionary containing the power factor under keys ['opamp']['power'].
    
    Returns:
    - float: The scaled derivative of the sigmoid function at the input value.
    """
    
    power = config["opamp"]["power"]
    return x * (1 - x) * power

def backpropagation(config: dict, output_data, target_data):
    """
    Perform the backpropagation algorithm for updating the weights of a neural network based on the
    error between the network's output and the target data.
    
    This function utilizes Numba's JIT compilation to optimize the execution speed, particularly 
    beneficial for the computational heavy loops involved in the backpropagation process.
    
    Parameters:
    - config (dict): A configuration dictionary containing the neural network's parameters such as 
                     weights and learning rate.
                     Expected keys are:
                     - 'simulation'['weights']: A list of numpy arrays representing the weights of each layer.
                     - 'learning'['learning_rate']: The learning rate used for weight updates.
    - output_data (list of numpy arrays): The activations of each layer of the network after forward propagation.
    - target_data (numpy array): The target output values for the network to learn.
    
    Returns:
    - list of numpy arrays: The updated weights after applying one backpropagation step.
    
    Notes:
    - The function assumes that sigmoid derivatives are used in the computation of deltas,
      and these are calculated separately in the sigmoid_derivative function.
    - The function edits the weights in-place to enhance memory usage efficiency.
    """
    weights = config["simulation"]["weights"]
    learning_rate = config["learning"]["learning_rate"]

    # Calculate the error and initial delta
    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1], config)

    # Preallocate new_weights for in-place computation
    new_weights = weights.copy()

    # Precompute sigmoid derivatives for all layers once if called multiple times
    sig_derivatives = [sigmoid_derivative(layer_output, config) for layer_output in output_data]

    # Iterate over layers in reverse order for backpropagation
    for layer in reversed(range(len(weights))):
        # Compute the gradients from the current delta and layer outputs
        gradients = output_data[layer].T.dot(delta)
        # Update weights in-place by subtracting the gradient scaled by the learning rate
        new_weights[layer] -= learning_rate * gradients

        if layer != 0:  # Avoid computation for the input layer where it's not needed
            # Update delta for the next layer to be processed
            delta = delta.dot(weights[layer].T) * sig_derivatives[layer - 1]

    return new_weights

def rprop(config: dict, output_data, target_data):
    """
    Perform the RPROP (Resilient Backpropagation) algorithm for updating neural network weights.

    This function modifies the configuration dictionary in-place to initialize and update
    parameters necessary for the RPROP algorithm. Changes to the configuration persist outside
    the function due to the mutable nature of dictionaries.

    Parameters:
    - config (dict): Configuration dictionary containing neural network settings and states.
    - output_data (list of np.ndarray): Activations from each layer of the network.
    - target_data (np.ndarray): Target output data for training.

    Returns:
    - list of np.ndarray: Updated weights after applying the RPROP algorithm.
    """
    weights = config["simulation"]["weights"]

    # Initialize RPROP parameters if not already set
    rprop_config = config.setdefault("learning", {}).setdefault("rprop", {})
    rprop_config.setdefault("delta_values", [0.1 * np.ones_like(w) for w in weights])
    rprop_config.setdefault("eta_plus", 1.2)
    rprop_config.setdefault("eta_minus", 0.5)
    rprop_config.setdefault("delta_max", 50)
    rprop_config.setdefault("delta_min", 1e-6)
    rprop_config.setdefault("last_gradients", [np.zeros_like(w) for w in weights])

    deltas = rprop_config["delta_values"]
    last_gradients = rprop_config["last_gradients"]

    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1], config)

    new_weights = weights.copy()

    for layer in reversed(range(len(weights))):
        gradients = output_data[layer].T.dot(delta)

        sign_change = np.sign(gradients) * np.sign(last_gradients[layer])
        increase_indices = sign_change > 0
        decrease_indices = sign_change < 0

        deltas[layer][increase_indices] = np.minimum(deltas[layer][increase_indices] * rprop_config["eta_plus"], rprop_config["delta_max"])
        deltas[layer][decrease_indices] = np.maximum(deltas[layer][decrease_indices] * rprop_config["eta_minus"], rprop_config["delta_min"])

        weight_update_indices = sign_change >= 0
        new_weights[layer][weight_update_indices] -= np.sign(gradients[weight_update_indices]) * deltas[layer][weight_update_indices]
        new_weights[layer][decrease_indices] = weights[layer][decrease_indices]

        last_gradients[layer] = gradients
        if layer > 0:
            delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer], config)

    return new_weights

def momentum(config: dict, output_data, target_data):
    """
    Apply the momentum update rule to optimize the neural network weights.

    This function updates the weights based on the gradient of the loss function, considering
    the previous velocity (momentum) to smooth out the updates.

    Parameters:
    - config (dict): Configuration dictionary containing weights, learning rate, and momentum settings.
    - output_data (list of np.ndarray): The activations for each layer from the forward pass.
    - target_data (np.ndarray): The actual target outputs for the training data.

    Returns:
    - list of np.ndarray: The updated weights after applying momentum.
    """
    weights = config["simulation"]["weights"]
    learning_rate = config["learning"]["learning_rate"]
    momentum_config = config["learning"]["momentum"]
    momentum_gamma = momentum_config["gamma"]

    # Initialize velocity if it does not exist
    if "velocity" not in momentum_config:
        momentum_config["velocity"] = [np.zeros_like(w) for w in weights]
    
    previous_velocity = momentum_config["velocity"]

    # Compute the error at the output layer
    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1], config)

    new_weights = []
    new_velocity = []

    for layer in reversed(range(len(weights))):
        gradients = output_data[layer].T.dot(delta)
        v = momentum_gamma * previous_velocity[layer] + learning_rate * gradients
        new_velocity.append(v)
        new_weights.append(weights[layer] - v)

        if layer != 0:
            delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer], config)
    
    # Update the velocities in the configuration
    config["learning"]["momentum"]["velocity"] = new_velocity[::-1]  # Reverse to maintain correct order

    return new_weights[::-1]  # Reverse the new_weights list to match the original order

def adam(config: dict, output_data, target_data):
    """
    Apply the Adam optimization algorithm to update the weights of a neural network.

    Parameters:
    - config (dict): Configuration dictionary containing the neural network settings and states.
    - output_data (list of np.ndarray): Activations from each layer of the network from the forward pass.
    - target_data (np.ndarray): Target outputs for the training data.

    Returns:
    - list of np.ndarray: Updated weights after applying the Adam optimization.
    """
    weights = config["simulation"]["weights"]
    learning_rate = config["learning"]["learning_rate"]
    adam_config = config["learning"].setdefault("adam", {})

    # Initialize Adam variables if they don't exist
    if "t" not in adam_config:
        adam_config["t"] = 0
        adam_config["mt"] = [np.zeros_like(w) for w in weights]
        adam_config["vt"] = [np.zeros_like(w) for w in weights]

    # Retrieve Adam parameters
    beta1 = adam_config.setdefault("beta1", 0.9)
    beta2 = adam_config.setdefault("beta2", 0.999)
    epsilon = adam_config.setdefault("epsilon", 1e-8)

    # Increment the timestep
    adam_config["t"] += 1
    t = adam_config["t"]
    mt = adam_config["mt"]
    vt = adam_config["vt"]

    error = target_data - output_data[-1]
    delta = error * sigmoid_derivative(output_data[-1], config)

    new_weights = []

    for layer in reversed(range(len(weights))):
        gradients = output_data[layer].T.dot(delta)

        # Update moment vectors
        mt[layer] = beta1 * mt[layer] + (1 - beta1) * gradients
        vt[layer] = beta2 * vt[layer] + (1 - beta2) * (gradients ** 2)

        # Compute bias-corrected moments
        mt_hat = mt[layer] / (1 - beta1 ** t)
        vt_hat = vt[layer] / (1 - beta2 ** t)

        # Update weights
        new_weights.append(weights[layer] - learning_rate * (mt_hat / (np.sqrt(vt_hat) + epsilon)))

        if layer > 0:
            delta = delta.dot(weights[layer].T) * sigmoid_derivative(output_data[layer], config)

    # Reverse the new_weights to maintain the original order
    return new_weights[::-1]
