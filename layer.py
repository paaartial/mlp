from neuron import Neuron

from helper import sigmoid, ReLu

class Layer:
    def __init__(self, l, af=sigmoid):
        self.length=l
        self.activation_function=af
        self.in_activations=[0 for n in range(l)]
        self.out_activations=[0 for n in range(l)]
        self.weight_matrix=[]
        self.biases=[]