import os
import sys
import numpy as np

from BackProp import Network, Layer, Neuron

NETWORK_DIRECTORY = os.path.join(os.getcwd(), 'Network_structures')

print("Welcome to BackProp simulation")
print("The network will be loaded from a JSON file, which you can provide")
print("Some sample testing JSON files are given, refer those to make your own custom network")

network = Network(os.path.join(NETWORK_DIRECTORY, 'network_1.json'))

train_x = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
train_y = np.array([1.0, 0.0])

network.forward_pass(train_x)
network.backward_pass(train_y)