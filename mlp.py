import random
import numpy as np
from numpy.core.fromnumeric import mean

from layer import Layer
from neuron import Neuron

from activation import sigmoid, sigmoid_prime
from cost import mean_squared_error, mse_prime

import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()
mnist_data_train, mnist_target_train, mnist_data_test, mnist_target_test = mnist[0][0], mnist[0][1], mnist[1][0], mnist[1][1]

class Net:
    def __init__(self, l, r):
        self.layers=[Layer(li) for li in l]
        self.input_layer = self.layers[0]
        self.learning_rate=r
        self.output_layer=self.layers[len(l)-1]
        for lay in range(1, len(self.layers)):
            for n in self.layers[lay].neurons:
                n.weights=[random.randint(-100, 100)/100 for pn in self.layers[lay-1].neurons]
    
    def feed_forward(self, input, target):
        self.input_layer.set_out_activations(input)
        for current_layer_index in range(1, len(self.layers)):
            wm=self.layers[current_layer_index].get_weight_matrix()
            a=self.layers[current_layer_index-1].get_out_activations()
            z=np.dot(wm, a)
            self.layers[current_layer_index].set_in_activations(z)
            self.layers[current_layer_index].set_out_activations(sigmoid(z))

        greatest_activation=0
        output_activations = self.output_layer.get_out_activations()
        for a in range(len(output_activations)):
            if output_activations[a]> output_activations[greatest_activation]:
                greatest_activation=a

        #print(output_activations)
        #print(greatest_activation)
        return {"error_prime" : mse_prime(self.output_layer.get_out_activations(), target), "prediction":greatest_activation, "error": mean_squared_error(self.output_layer.get_out_activations(), target)}



    def backpropogate(self, cost):
        # dC/dout * dout/din
        error_at_layer=[]
        sigmoid_prime_list = sigmoid_prime(self.output_layer.get_in_activations())
        error_at_layer.append([cost[n] * sigmoid_prime_list[n] for n in range(len(self.output_layer.neurons))])

        for l in range(len(self.layers)-2, 0, -1):
            #in     hidden1 hidden2     out
            weights_transpose = np.transpose(self.layers[l+1].get_weight_matrix())
            error_at_next_layer=error_at_layer[len(self.layers)-2-l]
            out_in=sigmoid_prime(self.layers[l].get_in_activations())
            z = np.dot(weights_transpose, error_at_next_layer)
            error_at_layer.append(z * out_in)
        # dout/din *din/dw

        #new_weights=[[[] for n in self.layers[i].neurons] for i in range(1, len(self.layers))]
        for l in range(len(self.layers)-1, 0, -1):
            current_layer=self.layers[l]
            for n in range(len(current_layer.neurons)):
                for w in range(len(current_layer.neurons[n].weights)):
                    """       |       use old weights for adjustments of new ones     """
                    """       V                                                       """
                                                                                  
                    new_weight = current_layer.neurons[n].weights[w] - self.learning_rate *error_at_layer[len(self.layers)-l-1][n] * self.layers[l-1].get_out_activations()[w]
                    self.layers[l].neurons[n].weights[w]=new_weight
                    #new_weights[l-1][n].append(new_weight)

        """for l in range(1, len(self.layers)-1):
            for n in range(len(self.layers[l].neurons)):
                self.layers[l].set_neuron_weights(new_weights[l-1][n], n)"""
        
                    

    def train(self, data, target, test_indices):
        for index in test_indices:
            print(index)
            to_feed=data[index].flatten()
            """for row in data[index]:
                for col in row:
                    to_feed.append(col/255)"""
            iteration = self.feed_forward(to_feed, target[index])
            self.backpropogate(iteration["error_prime"])
        return iteration
    
    def test(self, data, target, test_indices):
        average_error=0
        guess_percentage=0
        for index in test_indices:
            to_feed = data[index].flatten()
            """for row in data[index]:
                for col in row:
                    to_feed.append(col/255)"""
            iteration = self.feed_forward(to_feed, target[index])
            average_error+=sum(iteration["error"])
            if iteration["prediction"] == target[index]:
                guess_percentage +=1
        guess_percentage = guess_percentage / len(test_indices)
        average_error = average_error / len(test_indices)
        return {"%" : 100 * guess_percentage, "average error": average_error}

import skimage.measure

def conv(input):
    return skimage.measure.block_reduce(input, (1,3,3), np.max)

mnist_data_train = conv(mnist_data_train)
mnist_data_test = conv(mnist_data_test)
print(mnist_data_train.size)
test_net=Net([mnist_data_train[0].size, 128, 10], 0.05)

train_size=10000
test_size=1000
test_indices=[i for i in range(train_size, test_size+train_size)]
train_indices=[i for i in range(train_size)]

test_1 = test_net.test(mnist_data_train, mnist_target_train, test_indices)
print(test_1)

test_net.train(mnist_data_train, mnist_target_train, train_indices)

test_2 = test_net.test(mnist_data_train, mnist_target_train, test_indices)
print(test_2)

test_list=[1, 4, 2, 10, 5]