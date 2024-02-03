import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# from brilliant, "choose a random seed for reproducible results"
np.random.seed(1)

class layer:
    # constructor
    def __init__(self, n_neurons, n_inputs):
        #initialize data for weights [-1, 1]
        self.weights = 2 * np.random.rand(n_neurons, n_inputs + 1) - 1 # +1 for bias

    def print_weights(self):
        print("\n{}".format(self.weights))

    def feed_forward(self, input):
        input_with_bias = np.hstack((input, np.ones((input.shape[0], 1))))
        print("input with bias for layer: \n{}".format(input_with_bias))
        print()
        self.weighted_sum = np.dot(input_with_bias, self.weights.T)
        print("linear output of layer: \n{}".format(self.weighted_sum))
        print()
        #cache variable for backpropagation
        self.activated_output = sigmoid(self.weighted_sum)
        print("activated output of layer: \n{}".format(self.activated_output))
        print()
        return self.activated_output

    def calculate_output_error(self, target):
        self.layer_error = self.activated_output - target
        print("output error of layer: \n{}".format(self.layer_error))
        print()
        return self.layer_error

    def calculate_hidden_error(self, next_layer):
        sigmoid_derivative = deriv_sigmoid(self.weighted_sum)
        self.layer_error = sigmoid_derivative * (np.dot(next_layer.layer_error, next_layer.weights[:, 1:]))
        print("hidden error of layer: \n{}".format(self.layer_error))
        print()
        return self.layer_error

    def calculate_layer_gradient(self, input):
        self.layer_gradient = input[:,:, np.newaxis] * self.layer_error[:, np.newaxis, :]
        return self.layer_gradient
    
    def average_layer_gradient(self):
        self.layer_gradient_average = np.average(self.layer_gradient)
        print("layer's average gradient: \n{}".format(self.layer_gradient_average))
        print()
        return self.layer_gradient_average
    
    def update_weights(self, learning_rate):
        print("weights:\n{}".format(self.weights))
        print(" average gradient: \n{}".format(self.layer_gradient_average))
        self.weights += - learning_rate * self.layer_gradient_average
        
class net:
    def __init__(self, topology):
        self.layers = []
        for i in range(1, topology.size): # need to start from one to include the input layer
            self.layers.append(layer(topology[i], topology[i-1]))

    def print_weights(self):
        for i in range(len(self.layers)):
            print("Layer " + str(i+1) + " weights:\n{}".format(self.layers[i].weights))
            print() # newline

    def feed_forward(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].feed_forward(input)
        return input 

    def calculate_partials(self, input):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].calculate_layer_gradient(self.layers[i-1].activated_output)
            self.layers[0].calculate_layer_gradient(input)
            print("layer's partial derivative: \n{}".format(self.layers[i].layer_gradient))
            print()

        # handle the first hidden layer separately using the input data
        self.layers[0].calculate_layer_gradient(input)

    def average_gradients(self):
        for i in range(len(self.layers)):
            self.layers[i].average_layer_gradient()

    def update_weights(self, learning_rate):
        for i in range(len(self.layers)):
            self.layers[i].update_weights(learning_rate)
            
    def back_propagate(self, input, target, learning_rate):
        # calculate output error of last layer
        output_error = self.layers[-1].calculate_output_error(target)

        # loop through hidden layers and calculate error of hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].calculate_hidden_error(self.layers[i+1])

        # calculate partial derivatives of neural net layers
        self.calculate_partials(input)

        # calculate average for total gradients
        self.average_gradients()

        # update weights
        self.update_weights(learning_rate)

def main():
    # inputs
    X = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])

    # outputs
    y = np.array([[0, 1, 1, 0]]).T

    # defines structure of neural net
    topology = np.array([2, 3, 4, 1])
    
    num_epochs = 20000
    # neural net
    xor = net(topology)
    
    for i in range(num_epochs):
        print("epoch: " + str(i) + "\n")
        xor.feed_forward(X)
        xor.back_propagate(X, y, 0.15)
        print("Final output:\n{}".format(xor.layers[-1].activated_output))
        print()
        print("Expected output: \n{}".format(y))
        print("-----------------------------------")
        print()
    
# call main method to run program
main()
        
