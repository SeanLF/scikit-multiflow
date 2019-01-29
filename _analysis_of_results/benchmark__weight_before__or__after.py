import random
import numpy as np
from math import pi, e, sin, tanh

p = [random.random() for _ in range(3)]

def test1():
    y = [(1-tanh(3.5-7*pr))/2 for pr in p]
    return np.average(y)

def test2():
    return (1-tanh(3.5-7*np.average(p)))/2


if __name__ == '__main__':
    times = 10000000
    import timeit
    test1=timeit.timeit("test1()", setup="from __main__ import test1", number=(times))
    print(test1/times)
    test2=timeit.timeit("test2()", setup="from __main__ import test2", number=(times))
    print(test2/times)
    # test3=timeit.timeit("test3()", setup="from __main__ import test3", number=(1000000))
    # print(test3)

    print (test2/test1)