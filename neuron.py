import math
import numpy as np
from activation import *


class Neuron:
    
    def __init__(self, w=[], a=0, b=0):
        self.weights=w
        self.act=a
        self.bias=b
    
    def __repr__(self):
        return "act: " + str(self.act)