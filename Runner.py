import os
import sys
import numpy as np

from BackProp import NeuralNetwork, Layer, Neuron
from dataset_creator import dataset_creator

NETWORK_DIRECTORY = os.path.join(os.getcwd(), 'Network_structures')

print("Welcome to BackProp simulation")
print("The network will be loaded from a JSON file, which you can provide")
print("Some sample testing JSON files are given, refer those to make your own custom network")

network = NeuralNetwork(os.path.join(NETWORK_DIRECTORY, 'network_1.json'))

# Generate random data
# data = np.random.choice([0, 1], size=(10, 12), p=[0.5, 0.5])
# train_X, train_y = data[:, :10], data[:, 10:]

network.train_network()