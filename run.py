from json import load
from layer import Layer
from network import Network
import sys
import os

from helper import *

import skimage.measure

import numpy as np
import pandas as pd

np.random.seed(69)

kernel_size = (1,2,2)

def conv(input):
    return skimage.measure.block_reduce(input, kernel_size, np.max)

def load_network(network_name, new_name=""):
    if new_name=="":
        new_name=network_name
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = os.path.join("pretrained-networks", network_name +".json")
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path, 'r') as openfile:
        network_data = json.load(openfile)
        network_to_load = Network(new_name, network_data["layers"], network_data["lr"])
        network_to_load.set_weights(network_data["weights"]) 
    return network_to_load


if __name__ == "__main__":
    from tensorflow.keras.datasets.mnist import load_data

    mnist = load_data()
    mnist_train, mnist_test = (conv(mnist[0][0]), mnist[0][1]), (conv(mnist[1][0]), mnist[1][1])

    train_size = 10000
    test_size = 10000

    to_train, to_test = split_train_test(mnist_train, mnist_test, train_size+1, test_size)

    net = load_network("2x2Conv10000")
    net.test(to_test)
    #net = Network("train_complete", [mnist_train[0][0].size, 76, 10], 0.012)

    #first test:
    #start = 0.005, delta = 0.001, iterations = 50
    #train, test size = 10000

    #net = load_network("train_complete")
    #net.train_test_assess(to_train, to_test, intervals=[1000, 10000, 30000, 40000])

    """net.train(to_train)
    net.test(to_test)
    """

    """
    lrs = net.find_optimal_learning_rate(0.005, 0.001, 50, to_train, to_test)
    lrs_sorted = sort_things(lrs)
    print(lrs)
    print(lrs[lrs_sorted[0]]["%"])
    """


    num_epochs=1
    for epoch in range(num_epochs):
        net.train(to_train)
    net.test(to_test)
    net.save()

