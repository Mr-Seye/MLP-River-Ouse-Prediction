# This program will be making use of a neural network (specifically a
# multi-layered perceptron) to predict the flow of water through the gauging
# station located at Skelton of the river Ouse on 01/01/1997.

# Importing libraries to be used (EXCLUDING any neural libraries as specified
# in the specification)


import matplotlib
import numpy as np


# A class for the multi-layer perceptron


class Multi_Layer_Perceptron(object):

    # Contructor for the perceptron, which specifiy a number of inputs,
    # hidden layers and expected outputs.
    def __init__(self, n_inputs, h_layers, n_outputs):

        self.n_inputs = n_inputs
        self.h_layers = h_layers
        self.n_outputs = n_outputs

        # Creation of the layer represenations
        layers = [n_inputs] + h_layers + [n_outputs]

        # Randomise the inital weighting for each layer
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # Save the deriavtives for each layer
        deriv = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            deriv.append(d)
        self.deriv = deriv

        # Save the activations in each layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    # Function which computes the forward propagation of the neural network
    # based on the input
    # Arugments which are inputs return the activation values.
    def fwd_propagate(self, inputs):


        # The activation is the input passed through thus far
        activations = inputs

        # The activations are saved so that they can be used to
        # backwards propagate
        self.activations[0] = activations

        # The network then iterates through its layers
        for i, w in enumerate(self.weights):

            # Dot multiplcation between weights and previous
            # activation matricies
            net_inputs = np.dot(activations, w)

            # Application of the sigmoid actitvation function
            activations = self._sigmoid(net_inputs)

            # Save the new activations for backwards propagation
            self.activations[i + 1] = activations

        # Return activations to output layer
        return activations

    # Backwards propagation to match activations to the margin of error.
    # Taking the error to propagate and returning it as a final value for
    # the input
    def bck_propagate(self, error):

        # Iteration backwards though network layers
        for i in reversed(range(len(self.deriv))):

            # Get activation value for the previous layer
            activations = self.activations[i+1]

            # Application of the sigmoid deriavative function
            delta = error * self._sigmoid_derivative(activations)

            # Reshape (Transpose) the delta array to format it as a 2D array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # Get the activations for the current layer
            curr_activations = self.activations[i]

            # Reshape (Inverse) the activations matrix to format as a 2D matrix
            curr_activations = curr_activations.reshape(curr_activations.shape[0], -1)

            # Save derived derivative after application on matrix dot
            # multiplication
            self.deriv[i] = np.dot(curr_activations, delta_re)

            # Backward propagate the next error
            error = np.dot(delta, self.weights[i].T)

    # Training algorithim which takes input, target, epoch and learning 
    # rate values in order to train the multi-layer perceptron to make
    # a prediction on the dataset.
    def train(self, inputs, targets, epochs, learning_rate):

        # Beginning of the training iteration for the specified epochs
        for i in range(epochs):
            sum_errors = 0

            # Run though all of the provided training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # Forward propagate through the network
                output = self.fwd_propagate(input)

                # Determine the error margine
                error = target - output

                # Backward propagate the error back through the network
                self.bck_propagate(error)

                # Perform gradient descent on the derivatives in order to
                # update the previously randomly allocated weights
                self.gradient_descent(learning_rate)

                # Record the Mean Squared Error to report later on
                sum_errors += self._mse(target, output)

            # Completed epoch, report training error to the console and repeat
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")

    # Gradient descent function which dictates the
    # learning capability of the network
    def gradient_descent(self, learnRate):

        # Updating the weights by descending the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            deriv = self.deriv[i]
            weights += deriv * learnRate

    # Sigmoid activation function, passes an x value to be proecessed,
    # running it through the equation and providing a y result
    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    # Sigmoid derivative finds the gradient of the graph at x
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    # Mean Squared Error function measures the "loss", taking the
    # targetted value and actual output value and comparing them
    def _mse(self, target, output):
        return np.average((target - output) ** 2)


# Program is actually run here
if __name__ == "__main__":

    # Imported a cleaned dataset to train the network with

    items = np.genfromtxt("Training\[STANDARDISED] Ouse 93-94 Train Predictors.csv", delimiter=',')
    targets = np.genfromtxt("Training\[STANDARDISED] Ouse 93-94 Train Predictant.csv", delimiter=',')

    # Initialise the multi-layer perceptron passing the given parameters
    mlp = Multi_Layer_Perceptron(2, [3], 1)

    # Initiate the training of the network, providing number of epochs to be
    # run for and a learning rate
    mlp.train(items, targets, 50, 0.1)

    # Data to be verified against
    input = np.genfromtxt([20.7, 22.218])
    target = np.genfromtxt([63.81])

    # Initiate a forward propagation to receive a prediction
    output = mlp.fwd_propagate(input)

    # Reverse the standardisation of values
    output = output * (99.21 - 4.687) + 4.687

    print("The network predicts that the result of an average flow of {} and {} cubecs from the connecting streams will result in flow of {} cubecs at Skelton".format(input[0,0], input[0,1], output[0]))
    # print("Accuracy = {} %".format(target[0]/output[0]))
