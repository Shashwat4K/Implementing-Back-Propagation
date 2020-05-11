import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from network_config import *
from activation_functions import *

class Neuron(object):
    '''
    params: links: array of weights connected to the neuron from previous layer
            is_input: (boolean) tells whether 'this' neuron is an input neuron or not. 
            activation_type: Type of activation function
    '''
    def __init__(self, links, is_input=False, activation_type='sigmoid'):
        self.activation = 0.0
        self.is_input = is_input
        if is_input == False:
            self.links = links    
        if activation_type == 'sigmoid':
            self.activation_fn = sigmoid

    def get_activation(self):
        return self.activation

    def activate_neuron(self, prev_inputs):
        sum_value = np.dot(prev_inputs, self.links)
        activation_val = self.activation_fn(sum_value)
        self.activation = activation_val

class Layer(object):

    def __init__(self, num_units, layer_ID, layer_type, prev_layer_neurons):
        self.num_units = num_units 
        self.layer_type = layer_type
        self.prev_layer_neurons = prev_layer_neurons
        self.layer_ID = layer_ID

        # Initialize Neurons in the layer and, 
        # Initialize the weight matrix with random weights
        # The weight matrix is weights between 'this' and previous layer
         
        # For hidden and output layers
        if layer_type == 'h' or layer_type == 'o': 
            self.weight_matrix = np.random.uniform(low=-0.5, high=0.5, size=(prev_layer_neurons, num_units))
            self.neurons = [Neuron(self.weight_matrix[i]) for i in range(num_units)]
        # For input layer
        elif layer_type == 'i':
            self.weight_matrix = None 
            self.neurons = [Neuron(None, is_input=True) for i in range(num_units)]

    def get_weights(self):
        return self.weight_matrix

    def get_layer_ID(self):
        return self.layer_ID

    def get_weights_for_neuron(self, index):
        # return weight vector for neuron at index 'index'
        return self.weight_matrix[index]

    def update_weights(self):
        # Call this function only in backward pass.
        pass 

    def get_neuron_activations(self):
        return np.array([i.get_activation() for i in self.neurons])
        # returns Array of activations. shape = (n,)

    def get_neuron_count(self):
        return len(self.neurons)

class Network(object):
    
    def __init__(self):
        # Get network settings, from network config file
        self.layers = [] # Array of layers
        pass

    def forward_pass(self, input_vector):

        print("Forward pass: input='{}'".format(input_vector))
        # Input vector should be of same length as the number of neurons in input layer.
        assert len(input_vector)==self.layers[0].get_neuron_count()

        # For every layer after input layer,
        previous_layer_input = input_vector
        for layer in self.layers[1:]:
            temp = []
            # Calculate the weighted sum for every neuron
            for n in np.arange(layer.get_neuron_count()):
                # Calculate weighted sum (dot product) + add bias 
                weighted_sum = np.dot(previous_layer_input, layer.get_weights_for_neuron(n)) # + "add bias term here"
                # pass the weighted sum to the activation function
                layer.neurons[n].activate_neuron(weighted_sum)
                # accumulate the current activation values
                temp.append(layer.neurons[n].get_activation())

            # Update the previous layer input, which will be fed to the next layer
            previous_layer_input = np.array(temp) 
            temp.clear() 

    def backward_pass(self):
        pass 

    def calculate_error(self):
        pass

    # Train the network using back propagation algorithm
    # Use tqdm here!!! NOT IN FORWARD OR BACKWARD PASS!!!
    def train_network(self):
        pass

if __name__ == '__main__':
    print("Welcome to BackProp simulation")
    print("The network will be loaded from a JSON file, which you can provide")
    print("Some sample testing JSON files are given, refer those to make your own custom network")




