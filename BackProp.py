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
    def __init__(self, is_input=False, activation_type='sigmoid'):
        self.activation = 0.0
        self.is_input = is_input
        self.error_term = 0.0 # Delta for each neuron
        # self.links = "" # weights connecting 'this' neuron from all previous neurons, Empty for input neurons
        # if is_input == False:
        #     self.links = links    
        if activation_type == 'sigmoid':
            self.activation_fn = sigmoid

    def __str__(self):
        output_str = "<< Activation value: {}, Error term: {} >>".format(self.activation, self.error_term)
        return output_str

    def get_activation(self):
        return self.activation

    def set_error_term(self, value):
        self.error_term = value

    def get_error_term(self):
        return self.error_term    

    # prev_inputs: 'x', weights: 'w'. Take dot product of these two.
    def activate_neuron(self, prev_inputs, weights):
        # sum_value = np.dot(prev_inputs, self.links)
        sum_value = np.dot(prev_inputs, weights)
        activation_val = self.activation_fn(sum_value)
        self.activation = activation_val
    

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
            self.neurons = [Neuron() for i in range(self.num_units)]
        # For input layer
        elif self.layer_type == 'i':
            self.weight_matrix = None 
            self.neurons = [Neuron(is_input=True, activation_type='sigmoid') for i in range(self.num_units)]

    def get_weights(self):
        return self.weight_matrix

    def get_layer_ID(self):
        return self.layer_ID

    def get_layer_type(self):
        return self.layer_type    

    def get_weights_for_neuron(self, index):
        # return weight vector for neuron at index 'index'
        return self.weight_matrix[index]

    def get_neuron_activations(self):
        return np.array([n.get_activation() for n in self.neurons])
        # returns Array of activations. shape = (n,)

    def get_neuron_count(self):
        return len(self.neurons)

    def get_neuron_error_terms(self):
        return np.array([n.get_error_term() for n in self.neurons])

    def calculate_error_terms(self, is_output, resource):
        if is_output == True:
            # 'resource' is target output vector
            for n in range(len(self.neurons)):
                # delta = o * (1 - o) * (t - o)
                error_value = self.neurons[n].get_activation() * (1 - self.neurons[n].get_activation()) * (resource[n] - self.neurons[n].get_activation())
                self.neurons[n].set_error_term(error_value)
        else:
            # 'resource' is now weight matrix!
            (weight_matrix, error_vector) = resource
            for n in range(len(self.neurons)):
                temp_sum = np.dot(weight_matrix.T[n], error_vector)
                error_value = self.neurons[n].get_activation() * (1 - self.neurons[n].get_activation()) * temp_sum 
                self.neurons[n].set_error_term(error_value)

    # X is previous layer input, DELTA is vector of error terms
    def update_weights(self, X, DELTA, learning_rate):
        # Call this function only in backward pass.
        if self.layer_type != 'i':
            del_W = np.zeros(shape=self.weight_matrix.shape)
            for i in range(len(X)):
                for j in range(len(del_W)):
                    del_W[j,i] = learning_rate * X[i] * DELTA[j]
            self.weight_matrix = self.weight_matrix + del_W         


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

        # print("Forward pass: input='{}'".format(input_vector))
        # Input vector should be of same length as the number of neurons in input layer.
        assert len(input_vector)==self.layers[0].get_neuron_count()

        # For every layer after input layer,
        previous_layer_input = input_vector
        for layer in self.layers[1:]:
            temp = []
            # Calculate the weighted sum for every neuron
            for n in np.arange(layer.get_neuron_count()):
                # pass the required vectors (x, w) to the activate_neuron() method
                layer.neurons[n].activate_neuron(previous_layer_input, layer.get_weights_for_neuron(n))
                # accumulate the current activation values
                temp.append(layer.neurons[n].get_activation())

            # Update the previous layer input, which will be fed to the next layer
            previous_layer_input = np.array(temp) 
            temp.clear() 

    def backward_pass(self, target_output_vector):
        # l1 = self.layers[-1::-1]
        # l2 = self.layers[-2::-1]
        # l2.append(None)
        error_vector = None
        weight_matrix = None
        # Calculate Error terms for all layers (end to start):
        for current_layer in self.layers[::-1]:
            if current_layer.get_layer_type() == 'o':
                current_layer.calculate_error_terms(True, target_output_vector)
            elif current_layer.get_layer_type() == 'h':
                current_layer.calculate_error_terms(False, (weight_matrix, error_vector))
            else: 
                # If current layer is 'input layer' then stop processing
                break     
            error_vector = current_layer.get_neuron_error_terms()  
            weight_matrix = current_layer.get_weights()      
        
        # Update every weight now:
        previous_layer_neuron_activations = None
        for layer in self.layers:
            if layer.get_layer_type() != 'i':
                layer.update_weights(previous_layer_neuron_activations, layer.get_neuron_error_terms(), float(self.network_properties['learning_rate']))
            previous_layer_neuron_activations = layer.get_neuron_activations()    

    def calculate_error(self):
        pass

    # Train the network using back propagation algorithm
    # Use tqdm here!!! NOT IN FORWARD OR BACKWARD PASS!!!
    def train_network(self, training_data):
        X_train, y_train = training_data
        for i in tqdm(range(len(training_data))):
            self.forward_pass(X_train[i])
            self.backward_pass(y_train[i])

'''
if __name__ == '__main__':
    print("Welcome to BackProp simulation")
    print("The network will be loaded from a JSON file, which you can provide")
    print("Some sample testing JSON files are given, refer those to make your own custom network")

    CWD = os.getcwd()
    network = Network(os.path.join(CWD, 'Network_structures', 'network_1.json'))
'''