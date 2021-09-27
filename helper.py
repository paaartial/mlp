import math
import json

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

#COST
def mean_squared_error(predict, target):
    target_list=[0 for i in range(10)]
    target_list[target]=1
    try:
        assert len(predict) == len(target_list)
    except AssertionError:
        print("Target and prediction are not the same length")
    return [0.5 * (target_list[i]-predict[i])**2 for i in range(len(predict))]

def mse_prime(predict, target):
        target_list = [0 for i in range(10)]
        target_list[target] = 1
        return [a-t for (a, t) in zip(predict, target_list)]

#ACTIVATION
def sigmoid(xl, deriv=False):
    if not deriv:
        return [1/(1 + math.e**-x) for x in xl]
    return [sx * (1-sx) for sx in sigmoid(xl)]

def sigmoid_prime(xl):
    return [sx * (1-sx) for sx in sigmoid(xl)]

def ReLu(xl, deriv=False):
    if not deriv:
        return [max(x, 0) for x in xl]
    return [max(x, 0)/x for x in xl]
    
#DATASET MANIPULATIOM
def split_train_test(train, test, train_size, test_size):
    shuffled_train_indices = np.random.permutation(train_size)
    shuffled_test_indices = np.random.permutation(test_size)
    to_train_pairs = [(train[0][shuffled_index], train[1][shuffled_index]) for shuffled_index in range(len(shuffled_train_indices))]
    to_test_pairs = [(test[0][shuffled_index], test[1][shuffled_index]) for shuffled_index in range(len(shuffled_test_indices))]
    return (to_train_pairs, to_test_pairs)

#RANDOM SHIT
def draw_image(img_to_draw):
    some_digit_image = img_to_draw.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def dot_1d_transpose(l1, l2):
    rows=[]
    for r in range(len(l1)):
        rows.append([l1[r]*i for i in l2])
    return np.array(rows)


def element_wise_mult(l1, l2):
    try:
        assert len(l1) == len(l2)
    except:
        print("length of lists are not equal")

    return [l1[n]*l2[n] for n in range(len(l1))]