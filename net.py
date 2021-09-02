import random
import numpy as np
from activation import *


class Net:

    def __init__(self, *args):
        self.layers=args
        self.input_layer=args[0]
        self.hidden_layers=args[1:len(args)-1]
        self.output_layer=args[len(args)-1]
        #initializes network with random weights and biases for every neuron
        for layer_index in range(1, len(self.layers)):
            for n in self.layers[layer_index].neurons:
                n.weights=[random.randint(-100, 100)/100 for n in range(len(self.layers[layer_index-1]))]
                n.bias=random.randint(-100, 100)/100

    def output(self, img):
        for n in range(len(self.input_layer)):
            self.input_layer.neurons[n].act=img[n]/254
        for l in range(1, len(self.layers)):
            wm=self.layers[l].get_weight_matrix()
            acts=self.layers[l-1].get_activations()
            b=self.layers[l].get_biases()
            z=np.subtract(np.dot(wm, acts), b)
            if l == len(self.layers):
                self.layers[l].activate_neurons(sigmoid(z))
            else:
                self.layers[l].activate_neurons(ReLu(z))
        greatest_act_index=0
        output_acts=self.output_layer.get_activations()
        for act_index in range(len(output_acts)):
            if output_acts[act_index]>output_acts[greatest_act_index]:
                greatest_act_index=act_index
        print(greatest_act_index)
        #return output_acts

    def backpropogation(self):
        pass

    def train(self, train_set):
        pass

    def test(self, test_set):
        pass    