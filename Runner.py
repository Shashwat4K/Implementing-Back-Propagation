import os
import sys

from BackProp import Network, Layer, Neuron

NETWORK_DIRECTORY = os.path.join(os.getcwd(), 'Network_structures')

print("Welcome to BackProp simulation")
print("The network will be loaded from a JSON file, which you can provide")
print("Some sample testing JSON files are given, refer those to make your own custom network")

network = Network(os.path.join(NETWORK_DIRECTORY, 'network_1.json'))