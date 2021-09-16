from layer import Layer
from neuron import Neuron

from network import Net
from helper import *

import tensorflow as tf
import skimage.measure

import numpy as np
import pandas as pd

np.random.seed(42)

def conv(input):
    return skimage.measure.block_reduce(input, (1,3,3), np.max)

def load_network(network_name, new_name=""):
    if new_name=="":
        new_name=network_name
    with open(network_name +'.json', 'r') as openfile:
        network_data = json.load(openfile)
        network_to_load = Net(new_name, network_data["layers"], network_data["lr"])
        network_to_load.set_weights(network_data["weights"]) 
    return network_to_load

mnist = tf.keras.datasets.mnist.load_data()
mnist_train_data, mnist_train_target, mnist_test_data, mnist_test_target = conv(mnist[0][0]), mnist[0][1], conv(mnist[1][0]), mnist[1][1]
mnist_train, mnist_test = mnist[0], mnist[1]

train_size=100
test_size=100

to_train, to_test = split_train_test(mnist_train, mnist_test, train_size, test_size)
# ADD MORE WAYS TO EVALUATE NETWORK PERFORMANCE
# SPLIT TRAIN TEST
# FEED FORWARD, TRAIN, TEST TAKE TUPLE OF DATA AND TARGET

test_net=Net("test_net", [mnist_train_data[0].size, 128, 10], 0.05)

test_net.train(to_train)
test1=test_net.test(to_test)
print(test1)
#test_2 = nn.test(mnist_data_train, mnist_target_train, test_indices)
#print(test_2)

test_list=[1, 4, 2, 10, 5]