import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return 0 if x <= 0 else x

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return 0.1*x if x <= 0 else x 

def threshold(x, t=0):
    return 0 if x <= t else 1      