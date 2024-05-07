import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
mapfunspmax = np.log(1/0.001)
mapfunspmin = np.log(1/1)
spmin = 0.02
spmax = 0.07
# print(mapfunspmin, mapfunspmax, spmin, spmax)
# print(np.log())
def mapping_func(f):
    sp = np.log(1 / f)
    sp_mapped = (sp - mapfunspmin) * (spmax - spmin) / (mapfunspmax - mapfunspmin) + spmin # linear mapping
    return sp_mapped


x = np.linspace(2, 7, 100)
y = np.exp(x)
plt.figure(1)
plt.plot(x,1/y)

x = np.linspace(0.06, 1, 100)
y = np.log(1/x)
y_mapped = mapping_func(x)
plt.figure(2)
plt.plot(x, y)
plt.figure(3)
plt.plot(y_mapped)
plt.show()

