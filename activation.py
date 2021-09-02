import math
import numpy as np

def sigmoid(xl):
    return [1/(1 + math.e**(-x)) for x in xl]

def ReLu(xl):
    return [x * (1+(x/np.abs(x)))/2 for x in xl]