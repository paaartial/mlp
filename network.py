import random
import json
import time

import numpy as np
from numpy.core.fromnumeric import mean

from helper import ReLu, mse_prime, mean_squared_error, sigmoid, sigmoid_prime, element_wise_mult

from layer import Layer
from neuron import Neuron

class Net:
    def __init__(self, n, l, r):
        self.layers=[Layer(li) for li in l]
        self.input_layer = self.layers[0]
        self.learning_rate=r
        self.output_layer=self.layers[len(l)-1]
        self.name=n
        for lay in range(1, len(self.layers)):
            for n in self.layers[lay].neurons:
                n.weights = [random.randint(-30, 30)/100 for pn in self.layers[lay-1].neurons]
                n.bias = random.randint(-50, 50)/100
    
    def __repr__(self):
        return self.name

    def set_weights(self, lnw):
        for l in range(len(self.layers)):
            for n in range(len(self.layers[l].neurons)):
                self.layers[l].neurons[n].weights=lnw[l][n]

    def total_avg_error(self, pairs):
        err=[0 for i in range(10)]
        for pair in pairs:
            err = np.add(err, self.feed_forward(pair)["error_prime"])
        return [e/len(pairs) for e in err]

    def feed_forward(self, pair):
        self.input_layer.set_out_activations(pair[0].flatten())
        target = pair[1]
        for current_layer_index in range(1, len(self.layers)):
            cl = self.layers[current_layer_index]
            wm, a, b = cl.get_weight_matrix(), self.layers[current_layer_index-1].get_out_activations(), cl.get_biases()
            z=np.add(np.dot(wm, a), b)
            self.layers[current_layer_index].set_in_activations(z)
            self.layers[current_layer_index].set_out_activations(cl.activation_function(z))

        greatest_activation=0
        output_activations = self.output_layer.get_out_activations()
        for a in range(len(output_activations)):
            if output_activations[a]> output_activations[greatest_activation]:
                greatest_activation=a

        return {"error_prime" : mse_prime(self.output_layer.get_out_activations(), target), "prediction":greatest_activation, "error": mean_squared_error(self.output_layer.get_out_activations(), target)}

    def backpropogate(self, cost):
        # dC/dout * dout/din
        error_at_layer=[]
        sigmoid_prime_list = self.output_layer.activation_function(self.output_layer.get_in_activations(), deriv=True)
        error_at_layer.append([cost[n] * sigmoid_prime_list[n] for n in range(len(self.output_layer.neurons))])

        for l in range(len(self.layers)-2, 0, -1):
            #in     hidden1 hidden2     out
            weights_transpose = np.transpose(self.layers[l+1].get_weight_matrix())
            error_at_next_layer=error_at_layer[len(self.layers)-2-l]
            out_in=self.layers[l].activation_function(self.layers[l].get_in_activations(), deriv=True)
            z = np.dot(weights_transpose, error_at_next_layer)
            error_at_layer.append(z * out_in)
        # dout/din *din/dw

        for l in range(len(self.layers)-1, 0, -1):
            current_layer=self.layers[l]
            for n in range(len(current_layer.neurons)):
                new_bias = current_layer.neurons[n].bias - self.learning_rate *error_at_layer[len(self.layers)-l-1][n]
                self.layers[l].neurons[n].bias = new_bias
                for w in range(len(current_layer.neurons[n].weights)):                                                                  
                    new_weight = current_layer.neurons[n].weights[w] - self.learning_rate *error_at_layer[len(self.layers)-l-1][n] * self.layers[l-1].get_out_activations()[w]
                    self.layers[l].neurons[n].weights[w] = new_weight
        
    def train(self, train_pairs, track_progress=True):
        start_time = time.time()
        for train_index in range(len(train_pairs)):
            if track_progress:
                print("training: " + str(train_index) + "/" + str(len(train_pairs)))
            iteration = self.feed_forward(train_pairs[train_index])
            self.backpropogate(iteration["error_prime"])
        end_time = time.time()
        time_elapsed = end_time-start_time
        print("Time taken: " + str(time_elapsed//60) + " minutes, " + str(time_elapsed%60) + " seconds")
    
    def batch_gradient_descent(self, train_pairs):
        progress=1
        for pair in train_pairs:
            print(str(progress) + "/" + str(len(train_pairs)))
            total_error = self.total_avg_error(train_pairs)
            self.backpropogate(total_error)
            progress+=1
            
    def test(self, test_pairs, track_progress=False):
        average_error=0
        guess_percentage=0
        for test_index in range(len(test_pairs)):
            if track_progress:
                print("testing: " + str(test_index) + "/" + str(len(test_index)))
            iteration = self.feed_forward(test_pairs[test_index])
            average_error+=sum(iteration["error"])
            if iteration["prediction"] == test_pairs[test_index][1]:
                guess_percentage +=1
        guess_percentage = guess_percentage / len(test_pairs)
        average_error = average_error / len(test_pairs)
        return {"%" : 100 * guess_percentage, "average error": average_error}


    def save(self):
        with open(self.name +".json", "w") as outfile:
            to_save={}
            to_save["weights"]=[[n.weights for n in l.neurons] for l in self.layers]
            to_save["layers"]=[len(l.neurons) for l in self.layers]
            to_save["lr"] = self.learning_rate
            json.dump(to_save, outfile)