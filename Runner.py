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

network.train_network()
# network.predict_answer()