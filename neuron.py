class Neuron:
    def __init__(self, w=[], b=0, a1=0, a2=0):
        self.weights=w
        self.bias=b
        self.in_activation=a1
        self.out_activation=a2