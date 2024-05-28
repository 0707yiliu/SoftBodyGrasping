# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

time = np.arange(10)
temp = np.random.random(10)*30
Swdown = np.random.random(10)*100-10
Rn = np.random.random(10)*100-10

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, Swdown, '-', label = 'Swdown')
ax.plot(time, Rn, '-', label = 'Rn')
ax2 = ax.twinx()
ax2.plot(time, temp, '-r', label = 'temp')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("Time (h)")
ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
ax2.set_ylabel(r"Temperature ($^\circ$C)")
ax2.set_ylim(0, 35)
ax.set_ylim(-20,100)
ax2.legend(loc=0)
plt.show()