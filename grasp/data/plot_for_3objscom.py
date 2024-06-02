import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# collect data as list
datasets = [
    '20240519140659_0.05eggcooked_fixstep.npznew.npz', # G2
    '20240517142124_0.05eggcooked.npznew.npz', # G3
    '20240507154233_0.05eggcooked.npznew.npz', # G4
    '20240519161614_0.05chickenbreast_fixstep.npznew.npz', # G2
    '20240519161021_0.05chickenbreast.npznew.npz', # G3
    '20240519145807_0.05fish.npznew.npz', # G4
    '20240520174110_0.05mug_fixstep.npznew.npz', # G2
    '20240516165323_0.05mug.npznew.npz', # G3
    '20240516164233_0.05mug.npznew.npz',
]

y_data = [np.load(datasets[i])['y_data'] for i in range(len(datasets))]
z_data = [np.load(datasets[i])['z_data'] for i in range(len(datasets))]
ydz_data = [np.load(datasets[i])['ydz_data'] for i in range(len(datasets))]
dydz_data = [np.load(datasets[i])['dydz_data'] for i in range(len(datasets))]
des_slip_force = [np.load(datasets[i])['des_slip_force'] for i in range(len(datasets))]
des_slip_force[1] = des_slip_force[1][50:]
for i in range(len(datasets)):
    print('shapes:', i, y_data[i].shape, z_data[i].shape, ydz_data[i].shape, dydz_data[i].shape, des_slip_force[i].shape)

my_dpi=90
fig_w = 7.5
fig_h = 7.5
# ax2 = axs.twinx()
end = 1
lifttime = 40
datalen = y_data[0].shape[0]
limit_len = datalen - end
timeline = np.linspace(0, 40, limit_len)
fontsize=12
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : fontsize,
}
linestyle = ['-', '--', '-.', ':']
objs=[r'G2', r'G3', r'G4']
linewidth = 2
colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']


# fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h),dpi=my_dpi, sharex=False, sharey=False)
# for i in range(len(datasets)):
#     # line1 = axs.plot(y_data[i], linestyle=linestyle[0], color=colors[i], label = r'$f_y$', linewidth=linewidth)
#     # line2 = axs.plot(z_data[i], linestyle=linestyle[1], color=colors[i], label=r'$f_z$', linewidth=linewidth)
#     line3 = axs.plot(timeline, ydz_data[i][:-end], linestyle=linestyle[1], color=colors[i], label = r'$f_y/f_z$, '+objs[i], linewidth=linewidth)
#     line4 = axs.plot(timeline, dydz_data[i][:-end], linestyle=linestyle[0], color=colors[i], label=r'$\Delta f_y/\Delta f_z$, '+objs[i], linewidth=linewidth)
# axs.grid()
# axs.legend(loc=4, fontsize=fontsize-2, ncol=1)
# axs.set_xlabel("time (s)", fontsize=fontsize-2)
# axs.set_ylabel(r"Multi-Axis ($f_y/f_z$, $\Delta f_y/\Delta f_z$)", fontsize=fontsize-2)
# axs.tick_params(labelsize=fontsize-2)
# axs.set_xlim(0, lifttime)


fig, axs = plt.subplots(2, 3, figsize=(fig_w,fig_h),dpi=my_dpi, sharex=False, sharey=False)
hz = 80
timeline_recorder = np.zeros(12)
num_obj = 3
for j in range(num_obj): # 3objs
    init_timestep = 0
    for i in range(int(len(datasets)/num_obj)):
        print(i+j*3)
        datalen = y_data[i+j*3].shape[0]
        timestep = int(datalen / hz / 2)
        if timestep > init_timestep:
            init_timestep = timestep
        timeline_recorder[i+j*3] = timestep
        timeline = np.linspace(0, timestep, datalen-1)
        line1 = axs[0][j].plot(timeline, y_data[i+j*3][:-end], linestyle=linestyle[0], color=colors[i],  linewidth=linewidth)
        line2 = axs[0][j].plot(timeline, z_data[i+j*3][:-end], linestyle=linestyle[1], color=colors[i],  linewidth=linewidth)
    axs[0][j].grid()
axs[0][0].legend(['$f_y$,G2', '$f_z$,G2', '$f_y$,G3', '$f_z$,G3', '$f_y$,G4', '$f_z$,G4',], loc=3, fontsize=fontsize, ncol=3)
# axs[0][0].set_xlabel("time (s)", fontsize=fontsize)
axs[0][0].set_ylabel(r"Single-Axis ($f_y$, $f_z$)", fontsize=fontsize)
axs[0][0].tick_params(labelsize=fontsize)
axs[0][1].tick_params(labelsize=fontsize)
axs[0][2].tick_params(labelsize=fontsize)
axs[0][0].set_title(r"$id_{13}$", loc="center", fontsize=fontsize+2)
axs[0][1].set_title(r"$id_{21}$", loc="center", fontsize=fontsize+2)
axs[0][2].set_title(r"$id_{28}$", loc="center", fontsize=fontsize+2)
for j in range(num_obj): # 3objs
    init_timestep = 0
    for i in range(int(len(datasets) / num_obj)):
        print(i + j * 3)
        datalen = des_slip_force[i + j * 3].shape[0]
        timestep = int(datalen / hz)
        if timestep > init_timestep:
            init_timestep = timestep
        timeline = np.linspace(0, timeline_recorder[i+j*3], datalen - 1)
        line1 = axs[1][j].plot(timeline, des_slip_force[i + j * 3][:-end], linestyle=linestyle[0], color=colors[i],
                               linewidth=linewidth)
    axs[1][j].grid()
axs[1][0].legend(['G2', 'G3', 'G4',], loc=3, fontsize=fontsize, ncol=1)
axs[1][0].set_ylabel(r"$f_d$", fontsize=fontsize)
axs[1][0].set_xlabel(r"time (s)", fontsize=fontsize)
axs[1][1].set_xlabel(r"time (s)", fontsize=fontsize)
axs[1][2].set_xlabel(r"time (s)", fontsize=fontsize)
axs[1][0].tick_params(labelsize=fontsize)
axs[1][1].tick_params(labelsize=fontsize)
axs[1][2].tick_params(labelsize=fontsize)

plt.show()

