from layer import Layer
from neuron import Neuron
from network import Net

from activation import sigmoid, sigmoid_prime
from cost import mean_squared_error, mse_prime

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist.load_data()
mnist_data_train, mnist_target_train, mnist_data_test, mnist_target_test = mnist[0][0], mnist[0][1], mnist[1][0], mnist[1][1]

import skimage.measure

import json

def conv(input):
    return skimage.measure.block_reduce(input, (1,3,3), np.max)

def save_network(network_to_save):
    with open(str(network_to_save) +".json", "w") as outfile:
        to_save={}
        to_save["weights"]=[[n.weights for n in l.neurons] for l in network_to_save.layers]
        to_save["layers"]=[len(l.neurons) for l in network_to_save.layers]
        to_save["lr"] = network_to_save.learning_rate
        json.dump(to_save, outfile)
  
def load_network(network_name, new_name):
    with open(network_name +'.json', 'r') as openfile:
        network_data = json.load(openfile)
        network_to_load = Net(new_name, network_data["layers"], network_data["lr"])
        network_to_load.set_weights(network_data["weights"]) 
    return network_to_load
        
# ADD MORE WAYS TO EVALUATE NETWORK PERFORMANCE
# SPLIT TRAIN TEST

mnist_data_train = conv(mnist_data_train)
mnist_data_test = conv(mnist_data_test)
test_net=Net("test_net", [mnist_data_train[0].size, 128, 10], 0.05)

train_size=10000
test_size=10000
test_indices=[i for i in range(train_size, test_size+train_size)]
train_indices=[i for i in range(train_size)]

nn = load_network("test_net", "nn")
test_2 = nn.test(mnist_data_train, mnist_target_train, test_indices)
print(test_2)

test_list=[1, 4, 2, 10, 5]