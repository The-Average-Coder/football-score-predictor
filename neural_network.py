import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

        self.actual_values = [None] * len(layers)

        self.weights = [None] + [np.random.randn(layers[l], layers[l-1]) for l in range(1, len(layers))]
        self.biases = [None] + [np.random.randn(layers[l], 1) for l in range(1, len(layers))]

        self.final_layer = len(layers) - 1
        self.propagators = [None] * len(layers)

        self.dC_dW = [None] * len(layers)
        self.dC_db = [None] * len(layers)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def feed_forward(self, layer):
        if layer <= 0 or layer >= len(self.layers):
            return
        
        value = self.weights[layer] @ self.actual_values[layer-1] + self.biases[layer]
        self.actual_values[layer] = self.sigmoid(value)

    def calculate_cost(self, y_hat, y):
        losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

        m = y_hat.reshape(-1).shape[0]

        summed_losses = (1 / m) * np.sum(losses, axis=1)

        return np.sum(summed_losses)
    
    def back_propogation(self, l):
        if l <= 0 or l >= len(self.layers):
            return

        # Find the gradient of the cost with respect to the layer's node values
        dA_dZ = self.actual_values[l] * (1 - self.actual_values[l])
        dC_dZ = self.propagators[l] * dA_dZ

        # Find the gradients of the weights and biases
        dC_dW = dC_dZ @ self.actual_values[l-1].T
        dC_db = np.sum(dC_dZ, axis=1, keepdims=True)

        self.dC_dW[l] = dC_dW
        self.dC_db[l] = dC_db

        # Find the propagator term to continue the chain
        dC_dA_propagator = self.weights[l].T @ dC_dZ
        self.propagators[l-1] = dC_dA_propagator
    
    def train(self, training_data, expected_output, iterations, learning_rate):

        self.actual_values = [None] * len(self.layers)

        self.weights = [None] + [np.random.randn(self.layers[l], self.layers[l-1]) for l in range(1, len(self.layers))]
        self.biases = [None] + [np.random.randn(self.layers[l], 1) for l in range(1, len(self.layers))]

        self.final_layer = len(self.layers) - 1
        self.propagators = [None] * len(self.layers)

        self.dC_dW = [None] * len(self.layers)
        self.dC_db = [None] * len(self.layers)

        # Set input layer
        self.actual_values[0] = training_data.T

        costs = []

        for e in range(iterations):
            # Feed forward
            for l in range(1, len(self.layers)):
                self.feed_forward(l)

            y_hat = self.actual_values[self.final_layer]

            # Calculate cost
            cost = self.calculate_cost(y_hat, expected_output)
            costs.append(cost)

            # Backpropagation
            m = self.actual_values[self.final_layer].shape[1]
            self.propagators[self.final_layer] = (self.actual_values[self.final_layer] - expected_output).reshape(1, -1) / m
            
            for l in range(self.final_layer, 0, -1):
                self.back_propogation(l)

            # Update weights and biases
            for l in range(1, len(self.layers)):
                self.weights[l] -= learning_rate * self.dC_dW[l]
                self.biases[l] -= learning_rate * self.dC_db[l]

            if e % 20 == 0:
                print(f"epoch {e}: cost = {cost:4f}")
        
        return costs

    def predict(self, data):
        self.actual_values = [None] * len(self.layers)
        self.actual_values[0] = data.T

        for l in range(1, len(self.layers)):
                self.feed_forward(l)
        
        y_hat = self.actual_values[self.final_layer]

        return y_hat.flatten()