
def rprop_update(config: dict, output_data, target_data):
    weights = config["simulation"]["weights"]
    
    # Constants for RPROP
    eta_plus = 1.2   # Increase factor for the update value
    eta_minus = 0.4  # Decrease factor for the update value, slightly lower than the typical 0.5
    delta_max = 30   # Maximum step size, reduced from 50 to control the update magnitude
    delta_min = 0.00001  # Minimum step size, slightly higher than the typical very small value

    
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

