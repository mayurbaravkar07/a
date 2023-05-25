import numpy as np

# Parameters
vigilance_parameter = 0.5
learning_rate = 2

# Number of inputs
num_inputs = 4
num_categories = 3

# Initialize the bottom-up weights
bottom_up_weights = np.full((num_inputs, num_categories), 1 / (1 + num_inputs))

# Initialize the top-down weights
top_down_weights = np.ones((num_categories, num_inputs))

# ART-1 Network
def art1_network(input):
    reset = True
    while reset:
        reset = False
        si = input
        s_norm = sum(input)
        
        # Calculate the net input for each category
        y = np.dot(bottom_up_weights.T, input)
        
        # Find the winning category
        j = np.argmax(y)
        
        # Recompute the activation of F1 (bottom-up) layer for the winning category
        xi = si * top_down_weights[j]
        
        # Calculate x_norm
        x_norm = sum(xi)
        
        # Check the reset condition
        if x_norm / s_norm < vigilance_parameter:
            reset = True
        else:
            # Update the bottom-up weights for the winning category
            bottom_up_weights[:, j] = (learning_rate * xi) / (learning_rate - 1 + x_norm)
            # Update the top-down weights as well
            top_down_weights[j] = bottom_up_weights[:, j]
    
    return j

# Test the network
inputs = [[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]]

for i in inputs:
    print(art1_network(i))

