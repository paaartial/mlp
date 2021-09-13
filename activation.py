import math

def sigmoid(xl):
    return [1/(1 + math.e**-x) for x in xl]

def sigmoid_prime(xl):
    return [sx * (1-sx) for sx in sigmoid(xl)]