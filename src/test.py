import numpy as np

def backpropagate(config, output_data, target_data):
    weights = config["simulation"]["weights"]
    learning_rate = config["simulation"]["learning_rate"]
    
    # Calculate initial output layer delta
    # Assuming sigmoid activation function
    output_delta = (output_data[-1] - target_data) * output_data[-1] * (1 - output_data[-1])
    deltas = [output_delta]
    
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
                if np.ndim(gradient) > 0:
                    gradient = gradient.item()  # Convert numpy array to Python scalar
                updated_weight[r, c] -= learning_rate * gradient
        
        new_weights.append(updated_weight)
    
    return new_weights

# Configuration and dummy data for testing
config = {
    "simulation": {
        "geometry": [[36, 16], [16, 3]],
        "weights": [np.random.rand(36, 16), np.random.rand(16, 3)],
        "learning_rate": 0.01
    }
}

np.random.seed(42)
output_data = [np.random.rand(10, 36), np.random.rand(10, 16), np.random.rand(10, 3)]
target_data = np.random.rand(10, 3)

# Call the backpropagate function
new_weights = backpropagate(config, output_data, target_data)

# Print the average of the difference between the old and new weights
for old_weight, new_weight in zip(config["simulation"]["weights"], new_weights):
    print(np.mean(old_weight - new_weight))