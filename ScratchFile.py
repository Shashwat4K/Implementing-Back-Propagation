import numpy as np
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

def get_weight_and_error_vector_for_error_term_calculation(self, layer_type, neuron_number, next_layer_weight_matrix):
        return 1, 2

def calculate_delta_values(self, target_output_vector, next_layer_weight_matrix):
    # Calculate error term values for every neuron in this layer
    # To do that, we need weight_vec and error_vec (error_vec == None if layer is output layer)
    # TODO: Improvements HERE!!!!
    for i in range(len(target_output_vector)):
        weight_vec, error_vec = self.get_weight_and_error_vector_for_error_term_calculation(self.layer_type, i, next_layer_weight_matrix)

        self.neurons[i].calculate_error_term(self.layer_type, target_output_vector[i], weight_vec, error_vec)
            
def backward_pass(self, target_output_vector):
    # Calculate error term value for each neuron in the network, starting from the output layer all the way to input layer
    for layer in self.layers[::-1]:
        if layer.layer_type == 'o':
            next_layer_weight_matrix = None     
        if layer.layer_type != 'i': 
            layer.calculate_delta_values(target_output_vector, next_layer_weight_matrix)
            layer.update_weight_matrix()
            next_layer_weight_matrix = layer.get_weight_matrix()            