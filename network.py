import random
import json
import time

import numpy as np
from numpy.core.fromnumeric import mean

from helper import ReLu, mse_prime, mean_squared_error, sigmoid, element_wise_mult, dot_1d_transpose

from layer import Layer

class Net:
    def __init__(self, n, l, r):
        self.layers=[Layer(li) for li in l]
        self.input_layer = self.layers[0]
        self.learning_rate=r
        self.output_layer=self.layers[len(l)-1]
        self.name=n
        for lay in range(1, len(self.layers)):
            self.layers[lay].weight_matrix = [[random.randint(-30, 30)/100 for pn in range(self.layers[lay-1].length)] for n in range(self.layers[lay].length)]
            self.layers[lay].biases = [random.randint(-50, 50)/100 for n in range(self.layers[lay].length)]
    
    def __repr__(self):
        return self.name

    def set_weights(self, lnw):
        for l in range(len(self.layers)):
            self.layers[l].weight_matrix=lnw[l]

    def total_avg_error(self, pairs):
        err=[0 for i in range(10)]
        for pair in pairs:
            err = np.add(err, self.feed_forward(pair)["error_prime"])
        return [e/len(pairs) for e in err]

    def feed_forward(self, pair):
        self.input_layer.out_activations = pair[0].flatten()
        target = pair[1]
        for current_layer_index in range(1, len(self.layers)):
            cl = self.layers[current_layer_index]
            wm, a, b = cl.weight_matrix, self.layers[current_layer_index-1].out_activations, cl.biases
            z=np.add(np.dot(wm, a), b)
            self.layers[current_layer_index].in_activations = z
            self.layers[current_layer_index].out_activations = cl.activation_function(z)

        greatest_activation=0
        output_activations = self.output_layer.out_activations
        for a in range(len(output_activations)):
            if output_activations[a]> output_activations[greatest_activation]:
                greatest_activation=a

        return {"error_prime" : mse_prime(output_activations, target), "prediction" : greatest_activation, "error" : mean_squared_error(output_activations, target)}

    def backpropogate(self, cost):
        # dC/dout * dout/din
        error_at_layer=[]
        sigmoid_prime_list = self.output_layer.activation_function(self.output_layer.in_activations, deriv=True)
        error_at_layer.append([cost[n] * sigmoid_prime_list[n] for n in range(self.output_layer.length)])

        for l in range(len(self.layers)-2, 0, -1):
            #in     hidden1 hidden2     out
            weights_transpose = np.transpose(self.layers[l+1].weight_matrix)
            error_at_next_layer=error_at_layer[len(self.layers)-2-l]
            out_in=self.layers[l].activation_function(self.layers[l].in_activations, deriv=True)
            z = np.dot(weights_transpose, error_at_next_layer)
            error_at_layer.append(z * out_in)
        # dout/din *din/dw
        #print([len(error_at_layer[i]) for i in range(len(error_at_layer))])
        for l in range(len(self.layers)-1, 0, -1):
            #print(l)
            current_layer=self.layers[l]
            #print("current layer: " + str(l))
            #print(len(error_at_layer[len(self.layers)-l-1]))
            delta_w = dot_1d_transpose(error_at_layer[len(self.layers)-l-1], (self.layers[l-1].out_activations))
            self.layers[l].weight_matrix = np.subtract(self.layers[l].weight_matrix, self.learning_rate*delta_w)

            delta_b = np.array(error_at_layer[len(self.layers)-l-1])
            self.layers[l].biases = np.subtract(self.layers[l].biases, self.learning_rate * delta_b)
            """for n_index in range(current_layer.length):
                new_bias = current_layer.biases[n_index] - self.learning_rate *error_at_layer[len(self.layers)-l-1][n_index]
                self.layers[l].biases[n_index] = new_bias
                for n_index_prev in range(self.layers[l-1].length):                                                                  
                    new_weight = current_layer.weight_matrix[n_index][n_index_prev] - self.learning_rate *error_at_layer[len(self.layers)-l-1][n_index] * self.layers[l-1].out_activations[n_index_prev]
                    self.layers[l].weight_matrix[n_index][n_index_prev] = new_weight"""
        
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
            to_save["weights"]=[l.weight_matrix for l in self.layers]
            to_save["layers"]=[l.length for l in self.layers]
            to_save["lr"] = self.learning_rate
            json.dump(to_save, outfile)