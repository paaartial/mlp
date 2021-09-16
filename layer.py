from neuron import Neuron

class Layer:
    def __init__(self, l, af=None):
        self.length=l
        self.activation_function=af
        self.neurons=[Neuron() for n in range(l)]

    def get_biases(self):
        return [n.bias for n in self.neurons]

    def get_weight_matrix(self):
        return [n.weights for n in self.neurons]
    
    def set_neuron_weights(self, new_weights, n_index):
        self.neurons[n_index].weights=new_weights
        
    def get_in_activations(self):
        return [n.in_activation for n in self.neurons]

    def set_in_activations(self, a):
        for n in range(len(self.neurons)):
            self.neurons[n].in_activation=a[n]

    def get_out_activations(self):
        return [n.out_activation for n in self.neurons]

    def set_out_activations(self, a):
        for n in range(len(self.neurons)):
            self.neurons[n].out_activation=a[n]
