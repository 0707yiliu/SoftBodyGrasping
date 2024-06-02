import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# collect data as list
datasets = [
    '20240516151039_0.05kiwis.npznew.npz',
    # '20240519164011_0.05fish.npznew.npz',
    # '20240520161221_0.05cans.npznew.npz',
]

y_data = [np.load(datasets[i])['y_data'] for i in range(len(datasets))]
z_data = [np.load(datasets[i])['z_data'] for i in range(len(datasets))]
ydz_data = [np.load(datasets[i])['ydz_data'] for i in range(len(datasets))]
dydz_data = [np.load(datasets[i])['dydz_data'] for i in range(len(datasets))]
des_slip_force = [np.load(datasets[i])['des_slip_force'] for i in range(len(datasets))]
for i in range(len(datasets)):
    print('shapes:', y_data[i].shape, z_data[i].shape, ydz_data[i].shape, dydz_data[i].shape)
my_dpi=90
fig_w = 7.5
fig_h = 7.5
fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h),dpi=my_dpi, sharex=False, sharey=False)
# ax2 = axs.twinx()
end = 500
lifttime = 40
datalen = y_data[0].shape[0]
limit_len = datalen - end
timeline = np.linspace(0, 40, limit_len)
print(limit_len)
fontsize=20
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : fontsize,
}
linestyle = ['-', '--', '-.', ':']
objs=['kiwi', 'salmon', 'cans']
linewidth = 3
colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
for i in range(len(datasets)):
    # line1 = axs.plot(y_data[i], linestyle=linestyle[0], color=colors[i], label = r'$f_y$', linewidth=linewidth)
    # line2 = axs.plot(z_data[i], linestyle=linestyle[1], color=colors[i], label=r'$f_z$', linewidth=linewidth)
    print(len(ydz_data[i][:-end]))
    line3 = axs.plot(timeline, ydz_data[i][:-end], linestyle=linestyle[1], color=colors[i], label = r'$f_y/f_z$, '+objs[i], linewidth=linewidth)
    line4 = axs.plot(timeline, dydz_data[i][:-end], linestyle=linestyle[0], color=colors[i], label=r'$\Delta f_y/\Delta f_z$, '+objs[i], linewidth=linewidth)
axs.grid()
axs.legend(loc=4, fontsize=fontsize-2, ncol=1)
axs.set_xlabel("time (s)", fontsize=fontsize-2)
axs.set_ylabel(r"Multi-Axis ($f_y/f_z$, $\Delta f_y/\Delta f_z$)", fontsize=fontsize-2)
axs.tick_params(labelsize=fontsize-2)
axs.set_xlim(0, lifttime)


fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h),dpi=my_dpi, sharex=False, sharey=False)
for i in range(len(datasets)):
    line1 = axs.plot(timeline, y_data[i][:-end], linestyle=linestyle[0], color=colors[i], label = r'$f_y$, '+objs[i], linewidth=linewidth)
    line2 = axs.plot(timeline, z_data[i][:-end], linestyle=linestyle[1], color=colors[i], label=r'$f_z$, '+objs[i], linewidth=linewidth)
axs.grid()
axs.legend(loc=3, fontsize=fontsize-2, ncol=1)
axs.set_xlabel("time (s)", fontsize=fontsize-2)
axs.set_ylabel(r"Single-Axis ($f_y$, $f_z$)", fontsize=fontsize-2)
axs.tick_params(labelsize=fontsize-2)
axs.set_xlim(0, lifttime)

plt.show()

