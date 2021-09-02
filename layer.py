from neuron import Neuron

class Layer:
    
    def __init__(self, l):
        self.num_neurons=l
        self.neurons=[Neuron() for n in range(self.num_neurons)]

    def __len__(self):
        return self.num_neurons

    def __insert__(self, index, to_insert):
        self.neurons.insert(index, to_insert)

    def set_neuron(self, index, n):
        self.neurons[index]=n

    def activate_neurons(self, acts):
        for n in range(self.num_neurons):
            self.neurons[n].act=acts[n]

    def nudge_neuron_weight(self, index, weight, to_nudge):
        self.neurons[index].weights[weight]+=to_nudge

    def get_biases(self):
        biases=[n.bias for n in self.neurons]
        return biases

    def get_weight_matrix(self):
        weight_matrix=[n.weights for n in self.neurons]
        return weight_matrix

    def get_activations(self):
        activations = [n.act for n in self.neurons]
        return activations

    def __repr__(self):
        to_print=""
        for n in self.neurons:
            to_print+=str(n) + "\n"
        return to_print