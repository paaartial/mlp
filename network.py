import random
import math

import json
import time

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from layer import Layer

random.seed(42)

class Network:
    def __init__(self, n, l, r):
        self.layers=[Layer(li) for li in l]
        self.input_layer = self.layers[0]
        self.learning_rate=r
        self.output_layer=self.layers[len(l)-1]
        self.output_layer.activation_function=sigmoid
        self.name=n
        for lay in range(1, len(self.layers)):
            self.layers[lay].weight_matrix = [[random.randint(-30, 30)/100 for n in range(self.layers[lay-1].length)] for n2 in range(self.layers[lay].length)]
            self.layers[lay].biases = [random.randint(-50, 50)/100 for n in range(self.layers[lay].length)]

            #self.layers[lay].weight_matrix = np.random.rand(self.layers[lay].length, self.layers[lay-1].length)
            #self.layers[lay].biases = np.random.rand(self.layers[lay].length) 

            #limit = np.sqrt(6 / float(self.layers[lay-1].length + self.layers[lay].length))
            # scale = 1/max(1., (self.layers[lay].length+self.layers[lay].length)/2.)
            # limit = math.sqrt(3.0 * scale)
            # self.layers[lay].weight_matrix = np.random.uniform(low=-limit, high=limit, size=(self.layers[lay].length, self.layers[lay-1].length))
            # self.layers[lay].biases = np.zeros(self.layers[lay].length)
    
    def __repr__(self):
        return self.name

    #used for loading network. basically a better initialization method
    def set_weights(self, lnw):
        for l in range(len(self.layers)):
            self.layers[l].weight_matrix=lnw[l]

    #random is seeded, so this method just resets a neural network
    def clear_weights(self):
        for lay in range(1, len(self.layers)):
            self.layers[lay].weight_matrix = [[random.randint(-30, 30)/100 for n in range(self.layers[lay-1].length)] for n2 in range(self.layers[lay].length)]
            self.layers[lay].biases = [random.randint(-50, 50)/100 for n in range(self.layers[lay].length)]

    def total_avg_error(self, pairs):
        err=[0 for i in range(10)]
        for pair in pairs:
            err = np.add(err, self.feed_forward(pair)["error_prime"])
        return [e/len(pairs) for e in err]

    def feed_forward(self, pair):
        self.input_layer.out_activations = pair[0].flatten() / 255
        try:
            target = pair[1]
        except:
            pass
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
        try:
            return {
                "error_prime" : mse_prime(output_activations, target),
                "prediction" : greatest_activation, 
                "error" : mean_squared_error(output_activations, target)}
        except:
            return {
                "prediction" : greatest_activation
            }

    def backpropogate(self, cost):
        # dC/dout * dout/din
        error_at_layer=[]
        sigmoid_prime_list = self.output_layer.activation_function(self.output_layer.in_activations, deriv=True)
        error_at_layer.append([cost[n] * sigmoid_prime_list[n] for n in range(self.output_layer.length)])

        #loops over layers and calculates error
        for l in range(len(self.layers)-2, 0, -1):
            #in     hidden1 hidden2     out
            weights_transpose = np.transpose(self.layers[l+1].weight_matrix)
            error_at_next_layer=error_at_layer[len(self.layers)-2-l]
            out_in=self.layers[l].activation_function(self.layers[l].in_activations, deriv=True)
            z = np.dot(weights_transpose, error_at_next_layer)
            error_at_layer.append(z * out_in)
        # dout/din *din/dw

        #calculates weight, bias deltas from error
        for l in range(len(self.layers)-1, 0, -1):

            delta_w = dot_1d_transpose(error_at_layer[len(self.layers)-l-1], (self.layers[l-1].out_activations))
            self.layers[l].weight_matrix = np.subtract(self.layers[l].weight_matrix, self.learning_rate*delta_w)

            delta_b = np.array(error_at_layer[len(self.layers)-l-1])
            self.layers[l].biases = np.subtract(self.layers[l].biases, self.learning_rate * delta_b)
        
    def train_test_assess(self, train_pairs, test_pairs, intervals):
        x = []
        y = []
        for train_index in range(len(train_pairs)):
            if train_index in intervals or train_index==0 or train_index==len(train_pairs)-1:
                test = self.test(test_pairs)
                x.append(train_index)
                y.append(test["%"])
                print(str(train_index) + " : " + str(list(test.values())[0]))
            iteration = self.feed_forward(train_pairs[train_index])
            self.backpropogate(iteration["error_prime"])
        test = self.test(test_pairs)
        print(str(train_index) + " : " + str(list(test.values())[0]))
        plt.plot(x, y)
        plt.xlabel('train size')
        plt.ylabel('%' + " guess rate")
        plt.title('Performance by train size')
        plt.show()

    def train(self, train_pairs, track_progress=True):
        start_time = time.time()
        for train_index in range(len(train_pairs)):
            if track_progress:
                display_progress(train_index, len(train_pairs))
            iteration = self.feed_forward(train_pairs[train_index])
            self.backpropogate(iteration["error_prime"])
        end_time = time.time()
        time_elapsed = end_time-start_time
        #print("Time taken: " + str(time_elapsed//60) + " minutes, " + str(time_elapsed%60) + " seconds")

            
    def test(self, test_pairs, track_progress=False):
        guess_percentage=0
        for test_index in range(len(test_pairs)):
            if track_progress:
                print("why?")
            iteration = self.feed_forward(test_pairs[test_index])
            try:
                average_error=sum(iteration["error"])
            except KeyError:
                average_error = 0
            if iteration["prediction"] == test_pairs[test_index][1]:
                guess_percentage +=1
        guess_percentage = guess_percentage / len(test_pairs)
        average_error = average_error / len(test_pairs)
        if average_error:
            print("\n" + "stats: " + str({"%" : 100 * guess_percentage, "average error": average_error}))
            return {"%" : 100 * guess_percentage, "average error": average_error}
        else:
            print("\n" + "stats: " + str({"%" : 100 * guess_percentage}))
            return {"%" : 100 * guess_percentage}

    def find_optimal_learning_rate(self, start, delta, num_tests, train_pairs, test_pairs):
        performance_by_lr = {}
        total_train_pairs = num_tests * len(train_pairs)
        for lr_test_index in range(1, num_tests+1):
            self.learning_rate=start+delta*lr_test_index
            for train_index in range(len(train_pairs)):
                #display_progress(train_index + (lr_test_index-1) * len(train_pairs), total_train_pairs)
                print("test: " + str(lr_test_index) +"/" + str(num_tests) + " train progress: " + str(train_index) + "/" + str(len(train_pairs)))
                iteration = self.feed_forward(train_pairs[train_index])
                self.backpropogate(iteration["error_prime"])
            test=self.test(test_pairs)
            #print("test " + str(iteration) + " done")
            performance_by_lr[self.learning_rate] = test
            self.clear_weights()
        lr_best = performance_by_lr[sort_things(performance_by_lr)[0]]["%"]
        self.learning_rate = lr_best
        self.train(train_pairs)
        return performance_by_lr
        
    def save(self):
        import os
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = os.path.join("pretrained-networks", self.name +".json")
        abs_file_path = os.path.join(script_dir, rel_path)
        with open(abs_file_path, "w") as outfile:
            to_save={}
            to_save["weights"]=[[list(n) for n in l.weight_matrix] for l in self.layers]
            to_save["layers"]=[l.length for l in self.layers]
            to_save["lr"] = self.learning_rate
            json.dump(to_save, outfile)
            outfile.close()
    
    def batch_gradient_descent(self, train_pairs):
        progress=1
        for pair in train_pairs:
            print(str(progress) + "/" + str(len(train_pairs)))
            total_error = self.total_avg_error(train_pairs)
            self.backpropogate(total_error)
            progress+=1