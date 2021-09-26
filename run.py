from layer import Layer
from network import Net

from helper import *

from tensorflow.keras.datasets.mnist import load_data
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

mnist = load_data()
mnist_train, mnist_test = (conv(mnist[0][0]), mnist[0][1]), (conv(mnist[1][0]), mnist[1][1])

train_size = 10000
test_size = 1000

to_train, to_test = split_train_test(mnist_train, mnist_test, train_size, test_size)
test_net=Net("test_net", [mnist_train[0][0].size, 76, 10], 0.05)

test1=test_net.test(to_test)

test_net.train(to_train)
test2=test_net.test(to_test)
print(test1, test2)


#test_err = test_net.feed_forward(to_train[0])
#test_net.backpropogate(test_err)

test_list=[1, 4, 2, 10, 5]