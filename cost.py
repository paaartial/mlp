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