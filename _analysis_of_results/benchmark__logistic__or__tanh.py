import random
import numpy as np
from math import pi, e, sin, tanh

P = random.random()
SIGM = [14, 0.5]
TANH = [3.5, 7]

def _tanh_weight(x):
    return (1-tanh(TANH[0]-TANH[1]*x))/2

def _sigmoid_weight(x):
    return 1 / (1 + (np.exp((-SIGM[0]) * (x - SIGM[1]))))

def test1():
    return _tanh_weight(P)

def test2():
    return _sigmoid_weight(P)


if __name__ == '__main__':
    times = 10000000
    import timeit
    test1=timeit.timeit("test1()", setup="from __main__ import test1", number=(times))
    print('tanh:\t{}s'.format(test1/times))
    test2=timeit.timeit("test2()", setup="from __main__ import test2", number=(times))
    print('sigm:\t{}s'.format(test2/times))