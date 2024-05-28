import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    # print(window)
    re = np.convolve(interval, window, 'same')
    return re

def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs

tomatofix1 = np.array([0.31, 0.29, 0.33, 0.32, 0.29, 0.32, 0.35, 0.37])
tomato1 = np.array([0.291, 0.326, 0.314, 0.333, 0.298, 0.376, 0.354, 0.346])
tomatofix2 = np.array([0.14, 0.23, 0.25, 0.17, 0.14, 0.13, 0.19, 0.22])
tomato2 = np.array([0.173, 0.183, 0.215, 0.236, 0.264, 0.197, 0.275, 0.177])
kiwifix1 = np.array([0.3, 0.49, 0.41, 0.52, 0.69, 0.62, 0.45, 0.66, 0.55, 0.47])
kiwi1 = np.array([0.49,0.52, 0.69, 0.62, 0.45, 0.66, 0.55, 0.47])
orangefix1 = np.array([0.3, 0.49, 0.41, 0.52, 0.69, 0.62, 0.45, 0.66, 0.55, 0.47])
orange1 = np.array([0.983, 1.115, 1.037, 0.978, 1.013])

def meanandvar(data):

    mean = data.mean()
    var = data.var()
    print(mean, var)

meanandvar(orange1)