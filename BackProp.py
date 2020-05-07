import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from network_config import *

def sigmoid(arg1):
    return 'SOmething'

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
            self.weight_matrix = np.random.randn(prev_layer_neurons, num_units)
            self.neurons = [Neuron(self.weight_matrix[i]) for i in range(num_units)]
        # For input layer
        elif layer_type == 'i':
            self.weight_matrix = None 
            self.neurons = [Neuron(None, is_input=True) for i in range(num_units)]

    def get_weights(self):
        return self.weight_matrix

    def get_layer_ID(self):
        return self.layer_ID

    def update_weights(self):
        pass 

    def get_neuron_activations(self):
        return np.array([i.get_activation() for i in self.neurons])
        # returns Array of activations. shape = (n,)

class Network(object):

    def __init__(self):
        # Get network settings, from network config file
        pass     

