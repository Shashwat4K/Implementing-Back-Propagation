import os
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from activation_functions import sigmoid

class Neuron(object):
    '''
    params: links: array of weights connected to the neuron from previous layer
            is_input: (boolean) tells whether 'this' neuron is an input neuron or not. 
            activation_type: Type of activation function
    '''
    def __init__(self, links, is_input=False, activation_type='sigmoid'):
        self.activation = 0.0
        self.is_input = is_input
        self.error_term = 0.0 # Delta for each neuron
        self.links = "" # weights connecting 'this' neuron from all previous neurons, Empty for input neurons
        if is_input == False:
            self.links = links    
        if activation_type == 'sigmoid':
            self.activation_fn = sigmoid

    def __str__(self):
        output_str = "Activation value: {},\nError term: {}\nLinks: {}".format(self.activation, self.error_term, self.links)
        return output_str

    def get_activation(self):
        return self.activation

    def activate_neuron(self, prev_inputs):
        sum_value = np.dot(prev_inputs, self.links)
        activation_val = self.activation_fn(sum_value)
        self.activation = activation_val
    '''
    params:
        neuron_type: 'h', 'i', or 'o' suggests the type of neuron - hidden, input or output respectively.
        target_output: used only when neuron type is 'o', the target value at the particular output neuron
        weight_vec: Used to calculate error in hidden neuron. Vector of weights from 'this' neuron to all next layer neurons. 
        error_vec: Used to calculate error in hidden neuron. Vector of error terms of all next layer neurons
    '''
    def calculate_error_term(self, neuron_type, target_output, weight_vec, error_vec):
        if neuron_type == 'o':
            if target_output == None:
                raise 'Provide target output value to get error at output neuron!'
            else:
                self.error_term = self.activation * (1 - self.activation) * (target_output - self.activation)
        else:
            temp_sum = np.dot(weight_vec, error_vec)
            self.error_term = self.activation * (1 - self.activation) * temp_sum 

    def get_error_term(self):
        return self.error_term

class Layer(object):

    def __init__(self, layer_properties, prev_layer_neurons):
        self.num_units = layer_properties['neuron_count'] 
        self.layer_type = layer_properties['layer_type']
        self.layer_ID = layer_properties['layer_ID']
        self.activation_type = layer_properties['activation_type']
        self.layer_number = layer_properties['layer_number']
        self.prev_layer_neurons = prev_layer_neurons
        

        # Initialize Neurons in the layer and, 
        # Initialize the weight matrix with random weights
        # The weight matrix is weights between 'this' and previous layer
         
        # For hidden and output layers
        if self.layer_type == 'h' or self.layer_type == 'o': 
            self.weight_matrix = np.random.uniform(low=-0.5, high=0.5, size=(self.num_units, prev_layer_neurons))
            self.neurons = [Neuron(self.weight_matrix[i]) for i in range(self.num_units)]
        # For input layer
        elif self.layer_type == 'i':
            self.weight_matrix = None 
            self.neurons = [Neuron(None, is_input=True) for i in range(self.num_units)]

    def get_weights(self):
        return self.weight_matrix

    def get_layer_ID(self):
        return self.layer_ID

    def get_weights_for_neuron(self, index):
        # return weight vector for neuron at index 'index'
        return self.weight_matrix[index]

    def get_neuron_activations(self):
        return np.array([i.get_activation() for i in self.neurons])
        # returns Array of activations. shape = (n,)

    def get_neuron_count(self):
        return len(self.neurons)

    def get_weight_and_error_vector_for_error_term_calculation(self, layer_type):
        pass

    def calculate_delta_values(self, target_output_vector):
        # Calculate error term values for every neuron in this layer
        # To do that, we need weight_vec and error_vec (error_vec == None if layer is output layer)
        # TODO: Improvements HERE!!!!
        for i in range(len(target_output_vector)):
            weight_vec, error_vec = self.get_weight_and_error_vector_for_error_term_calculation(self.layer_type)

            self.neurons[i].calculate_error_term(self.layer_type, target_output_vector[i], weight_vec, error_vec)
            

    def update_weight_matrix(self):
        # Call this function only in backward pass.
        pass 


    def print_layer_properties(self):
        print('**************')
        print('Layer Number: {}\nLayer ID: {}\nLayer Type: {}\nNeuron count: {}\nActivation type: {}'.format(
            self.layer_number,
            self.layer_ID,
            self.layer_type,
            self.num_units,
            self.activation_type
        )) 
        print('**************')

    def print_layer_neurons(self):
        for i in range(self.num_units):
            print('{})'.format(i+1))
            print(self.neurons[i])
            print('****************')  

class Network(object):
    
    def __init__(self, network_file_path):
        # Get network settings, from network config file
        with open(network_file_path, "r") as net:
            self.network_properties = json.load(net)
        self.layers = [] # Array of layers
        prev_layer_units = None
        for i in range(self.network_properties['n_layers']+1):
            current_layer = self.network_properties['layers'][i]
            self.layers.append(Layer(current_layer, prev_layer_units))
            prev_layer_units = current_layer['neuron_count']

        self.print_layers()
        
    def print_layers(self):
        print('printing all the layers:')
        for i in range(self.network_properties['n_layers']+1):
            self.layers[i].print_layer_properties()
            print('#################')

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

    def backward_pass(self, target_output_vector):
        # Calculate error term value for each neuron in the network, starting from the output layer all the way to input layer
        for layer in self.layers[::-1]:
            if layer.layer_type != 'i': 
                layer.calculate_delta_values(target_output_vector)
                layer.update_weight_matrix()

    def calculate_error(self):
        pass

    # Train the network using back propagation algorithm
    # Use tqdm here!!! NOT IN FORWARD OR BACKWARD PASS!!!
    def train_network(self):
        pass

'''
if __name__ == '__main__':
    print("Welcome to BackProp simulation")
    print("The network will be loaded from a JSON file, which you can provide")
    print("Some sample testing JSON files are given, refer those to make your own custom network")

    CWD = os.getcwd()
    network = Network(os.path.join(CWD, 'Network_structures', 'network_1.json'))
'''