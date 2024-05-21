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

# collect data as list
datasets = [
    # '20240516144816_0.05mug_basicfordet.npz',
    '20240516145034_0.05kiwis_basicfordet.npz',
    # '20240411141416_0.08force_cup_lift1-0.005-0.005.npz',
    # '20240412112624_1.3force_cup_lift1-0.005-0.005.npz',
    # '20240412115404_1.5force_cup_lift1-0.005-0.005.npz',
    # '20240412120426_1.8force_cup_lift1-0.005-0.005.npz',
    # '20240412150851_3force_cup_lift1-0.005-0.005.npz',
]
tac_datalists = [np.load(datasets[i])['loop_tac_data'] for i in range(len(datasets))]
pos_datalists = [np.load(datasets[i])['gripper_pos'] for i in range(len(datasets))]
#- -------------------11111111-----------------------------
all_tac_data = tac_datalists[0]
#- -------------------222222222-----new------------------------
all_tac_dataALL = [np.load(datasets[i])['all_tac_data'] for i in range(len(datasets))][0]
des_slip_force = [np.load(datasets[i])['des_slip_force'] for i in range(len(datasets))][0]
# print(all_tac_data)
print('loop shape and all shape:',all_tac_data.shape, all_tac_dataALL.shape)
#- ------------------------------------------------
for i in range(all_tac_dataALL.shape[1]):
    all_tac_dataALL[:, i] = moving_average(all_tac_dataALL[:, i], 10)
for i in range(all_tac_data.shape[1]):
    all_tac_data[:, i] = moving_average(all_tac_data[:, i], 10)
# print('----------', all_tac_data.shape)
d_all_tac_data = FirstOrderLag(all_tac_dataALL, 0.8)
all_tac_dataALL = FirstOrderLag(all_tac_dataALL, 0.8)
all_tac_data = FirstOrderLag(all_tac_data, 0.7)

hz_time = 0.02
hz = 1 / hz_time
# print('---', all_tac_data.shape)
d_all_tac_data = np.vstack((np.zeros(all_tac_data.shape[1]), d_all_tac_data))
for i in range(d_all_tac_data.shape[0]-1):
    d_all_tac_data[i, :] = (d_all_tac_data[i+1, :] - d_all_tac_data[i,:]) / hz_time
d_all_tac_data = np.delete(d_all_tac_data, -1, 0)
# for i in range(d_all_tac_data.shape[0]):
#     d_all_tac_data[i, :] = moving_average(d_all_tac_data[i, :], 10)

for i in range(len(datasets)):
    if len(tac_datalists[i]) != len(pos_datalists[i]):
        pos_datalists[i] = np.delete(pos_datalists[i], 0, 0)
    print(len(tac_datalists[i]), len(pos_datalists[i]))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
legends = ['bottle', 'chips', 'orange', 'tomato', 'bread', 'eggshell', 'wholebanana', 'banana']

dataall = np.load('20240315150348_banana.npz')
print(dataall['loop_tac_data'].shape)
tac_data = dataall['loop_tac_data']
data_xlen = tac_data.shape[0]
data_xlen = np.linspace(0, data_xlen-1, data_xlen)
# print(data_xlen, data_xlen.shape)
pos_data = dataall['gripper_pos']
# pos_data = np.delete(pos_data, 0, 0)
pos_data_len = pos_data.shape[0]
# print(pos_data_len)

# fig1 = plt.figure(1)
# plt.plot(np.linspace(0, pos_data_len-1, pos_data_len), pos_data)

my_dpi=90
row = 2
col = 6
xlabel = 'time'
ylabel = [
            'sensor1_x', 'sensor1_y', 'sensor1_z',
            'sensor2_x', 'sensor2_y', 'sensor2_z',
            'sensor3_x', 'sensor3_y', 'sensor3_z',
            'sensor4_x', 'sensor4_y', 'sensor4_z',
]
fig, axs = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
_legend = False
# print(np.linspace(0, len(tac_datalists[0])-1, len(tac_datalists[0])))
# print(tac_datalists)
stay_item = 1
fig.suptitle(datasets[0], fontsize=20)
legends = ['squeeze', 'hold', 'lift']
# for cut
head = 10
end = 10
fig1tac = np.delete(tac_datalists[0], np.arange(head).tolist(), 0)
fig1tac = np.delete(fig1tac, (np.arange(end)*-1).tolist(), 0)
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        for k in range(len(datasets)):
            axs[i][j].plot(np.linspace(0,
                                       len(fig1tac)-1,
                                       len(fig1tac)),
                           fig1tac[:, yl],
                           color=colors[k], label=legends[k])
        if _legend is False:
            axs[i][j].legend()
            _legend = True
        axs[i][j].legend()
        # axs[i][j].plot(data_xlen, tac_data[:, i+j])
        axs[i][j].set_ylabel(ylabel[yl])
        axs[i][j].set_xlabel('time')

fig1, axs1 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
_legend = False
legends = ['origin']
fig1.suptitle('all tac data', fontsize=20)
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        # axs1[i][j].plot(np.linspace(0,
        #                             d_all_tac_data.shape[0]-1,
        #                             d_all_tac_data.shape[0]),
        #                 d_all_tac_data[:, yl],
        #                 color=colors[0], label=legends[k])
        axs1[i][j].plot(np.linspace(0,
                                    all_tac_dataALL.shape[0] - 1,
                                    all_tac_dataALL.shape[0]),
                        all_tac_dataALL[:, yl],
                        color=colors[0], label=legends[k])

        if _legend is False:
            axs1[i][j].legend()
            _legend = True
        # axs1[i][j].plot(pos_data, tac_data[:, i+j])
        axs1[i][j].set_ylabel(ylabel[yl])
        axs1[i][j].set_xlabel('time')



from matplotlib.collections import LineCollection
from matplotlib import cm
import copy
import matplotlib.colors as mcolors
def color_map(data, cmap):
    """数值映射为颜色"""

    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256 / cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255 * (data - dmin) / (dmax - dmin))

    return cs[data]

row = 2
col = 2
xmin = -0.3
xmax = 0.1
ymin = -0.3
ymax = 0
y_force = ['sensro1_Y', 'sensro2_Y','sensro3_Y','sensro4_Y',]
fig2, axs2 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
fig2.suptitle('force z-y', fontsize=20)
_legend = False
for i in range(row):
    for j in range(col):
        index = (i * 2 + j) + 1
        yl = index * 3 - 1
        # print(index, yl)
        # print(all_tac_data.shape, np.arange(0, 1, 0.1), np.linspace(0,1,11))
        ax = axs2[i][j].scatter(all_tac_dataALL[:, yl-1],
                                all_tac_dataALL[:, yl-0],
                                c=np.linspace(0, 1, num=all_tac_dataALL.shape[0]),
                                cmap='spring', vmin=0, vmax=0.35)
        fig2.colorbar(ax, orientation='horizontal')
        # for num in range(all_tac_data.shape[0]):
        #     axs2[i][j].plot(all_tac_data[num, yl],
        #                     all_tac_data[num, yl-1], 'o', c=cm.OrRd(num/all_tac_data.shape[0]))
        axs2[i][j].plot(all_tac_dataALL[:, yl-1],
                        all_tac_dataALL[:, yl-0])
        if _legend is False:
            axs2[i][j].legend()

            _legend = True
        # axs1[i][j].plot(pos_data, tac_data[:, i+j])
        # axs2[i][j].set_xlim(xmin, xmax)
        # axs2[i][j].set_ylim(ymin, ymax)
        axs2[i][j].set_ylabel('Z')
        axs2[i][j].set_xlabel(y_force[index-1])



# row = 2
# col = 2
# x_force = ['sensro1_X', 'sensro2_X', 'sensro3_X', 'sensro4_X', ]
# fig2, axs2 = plt.subplots(row, col, figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi, sharex=False,
#                           sharey=False)
# fig2.suptitle('force x-z', fontsize=20)
# _legend = False
# for i in range(row):
#     for j in range(col):
#         index = (i * 2 + j) + 1
#         yl = index * 3 - 1
#         # print(index, yl)
#         # print(all_tac_data.shape, np.arange(0, 1, 0.1), np.linspace(0,1,11))
#         ax = axs2[i][j].scatter(all_tac_data[:, yl - 2],
#                                 all_tac_data[:, yl],
#                                 c=np.linspace(0, 1, num=all_tac_data.shape[0]),
#                                 cmap='spring', vmin=0, vmax=0.35)
#         fig2.colorbar(ax, orientation='horizontal')
#         # for num in range(all_tac_data.shape[0]):
#         #     axs2[i][j].plot(all_tac_data[num, yl],
#         #                     all_tac_data[num, yl-1], 'o', c=cm.OrRd(num/all_tac_data.shape[0]))
#         axs2[i][j].plot(all_tac_data[:, yl - 2],
#                         all_tac_data[:, yl])
#         if _legend is False:
#             axs2[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs2[i][j].set_ylabel('Z')
#         axs2[i][j].set_xlabel(x_force[index-1])
#
# row = 2
# col = 2
# x_force = ['sensro1_X', 'sensro2_X', 'sensro3_X', 'sensro4_X', ]
# y_force = ['sensro1_Y', 'sensro2_Y','sensro3_Y','sensro4_Y',]
# fig2, axs2 = plt.subplots(row, col, figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi, sharex=False,
#                           sharey=False)
# fig2.suptitle('force x-y', fontsize=20)
# _legend = False
# for i in range(row):
#     for j in range(col):
#         index = (i * 2 + j) + 1
#         yl = index * 3 - 1
#         # print(index, yl)
#         # print(all_tac_data.shape, np.arange(0, 1, 0.1), np.linspace(0,1,11))
#         ax = axs2[i][j].scatter(all_tac_data[:, yl - 2],
#                                 all_tac_data[:, yl - 1],
#                                 c=np.linspace(0, 1, num=all_tac_data.shape[0]),
#                                 cmap='spring', vmin=0, vmax=0.35)
#         fig2.colorbar(ax, orientation='horizontal')
#         # for num in range(all_tac_data.shape[0]):
#         #     axs2[i][j].plot(all_tac_data[num, yl],
#         #                     all_tac_data[num, yl-1], 'o', c=cm.OrRd(num/all_tac_data.shape[0]))
#         axs2[i][j].plot(all_tac_data[:, yl - 2],
#                         all_tac_data[:, yl - 1])
#         if _legend is False:
#             axs2[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs2[i][j].set_ylabel(y_force[index-1])
#         axs2[i][j].set_xlabel(x_force[index-1])
#
# row = 2
# col = 2
# xmin = -0.5
# xmax = 40
# ymin = -0
# ymax = 130
# x_force = ['sensro1_X', 'sensro2_X', 'sensro3_X', 'sensro4_X', ]
# y_force = ['sensro1_Y', 'sensro2_Y', 'sensro3_Y', 'sensro4_Y', ]
# d_zy = ['sensro1_ZY', 'sensro2_ZY', 'sensro3_ZY', 'sensro4_ZY', ]
# fig2, axs2 = plt.subplots(row, col, figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi, sharex=False,
#                           sharey=False)
# fig2.suptitle('force integral z-y', fontsize=20)
# num_all_tac = all_tac_data.shape[0]
# _len = 50 # time delay is 0.02 (50hz)
# step_num = num_all_tac // _len
# left_num = num_all_tac % _len
# if left_num != 0:
#     step_num += 1
# d_all_tac_data_zy = np.zeros((step_num - 1, 4))
# z_index = [2, 5, 8, 11]
# # print(all_tac_data[:, 0])
# for i in range(step_num-1):
#     for j in range(len(z_index)):
#         if i > 0:
#             d_all_tac_data_zy[i, j] = (abs(all_tac_data[(i + 1) * _len, z_index[j]] - all_tac_data[i * _len, z_index[j]])) * (abs(
#                 all_tac_data[(i + 1) * _len, z_index[j] - 1] - all_tac_data[i * _len, z_index[j] - 1])) + d_all_tac_data_zy[i - 1, j]
#         else:
#             d_all_tac_data_zy[i, j] = (abs(all_tac_data[(i + 1) * _len, z_index[j]] - all_tac_data[i * _len, z_index[j]])) * (abs(
#                 all_tac_data[(i + 1) * _len, z_index[j] - 1] - all_tac_data[i * _len, z_index[j] - 1]))
#         # print((abs(all_tac_data[(i + 1) * _len, z_index[j]] - all_tac_data[i * _len, z_index[j]])),
#         #       (abs(all_tac_data[(i + 1) * _len, z_index[j] - 1] - all_tac_data[i * _len, z_index[j] - 1])),
#         #       d_all_tac_data_zy[i], z_index[j])
#         # time.sleep(100)
# print(d_all_tac_data_zy.shape)
# _legend = False
# for i in range(row):
#     for j in range(col):
#         index = (i * 2 + j) + 1
#         axs2[i][j].plot(np.linspace(0,
#                                     d_all_tac_data_zy.shape[0] - 1,
#                                     d_all_tac_data_zy.shape[0]),
#                         d_all_tac_data_zy[:, index - 1])
#         # axs2[i][j].plot(all_tac_data[:, 0])
#         if _legend is False:
#             axs2[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs2[i][j].set_xlim(xmin, xmax)
#         axs2[i][j].set_ylim(ymin, ymax)
#         axs2[i][j].set_ylabel(d_zy[index - 1])
#         axs2[i][j].set_xlabel('time')

row = 2
col = 2
xmin = -0.3
xmax = 0.1
ymin = -0.3
ymax = 0
y_force = ['sensro1_Y/Z', 'sensro2_Y/Z','sensro3_Y/Z','sensro4_Y/Z',]
fig2, axs2 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
fig2.suptitle('force y/z', fontsize=20)
_legend = False
y_z_tac_data = np.zeros((all_tac_data.shape[0], 4))
dy_z_tac_data = np.zeros((all_tac_data.shape[0], 4))
for i in range(row):
    for j in range(col):
        index = (i * 2 + j) + 1
        yl = index * 3 - 1
        # print(index, yl, y_z_tac_data.shape, all_tac_data.shape)
        y_z_tac_data[:, index-1] = all_tac_data[:, yl-1] / all_tac_data[:, yl]
# print(y_z_tac_data.shape[0])
# derivation delta-yz
_hz_step = 5
step_num = y_z_tac_data.shape[0] // _hz_step
dy_z_tac_data = np.zeros((step_num, 4))
dy_z_tac_data_inter = np.zeros((y_z_tac_data.shape[0], 4))
print(dy_z_tac_data_inter.shape)
for i in range(step_num):
    # print(i)
    if i == 0:
        dy_z_tac_data[i, :] = (y_z_tac_data[i * _hz_step, :]) / (0.02 * _hz_step)
    else:
        dy_z_tac_data[i, :] = (y_z_tac_data[i * _hz_step, :] - y_z_tac_data[(i - 1) * _hz_step, :]) / (0.02 * _hz_step)
# print(all_tac_data.shape)
# for interp
LEN = y_z_tac_data.shape[0]
for i in range(4):
    xlen = dy_z_tac_data.shape[0]
    print('xlen for dyztac:', xlen, dy_z_tac_data[:, i].shape)
    xold = np.linspace(0,LEN-1, num=xlen)
    xnew = np.linspace(0,LEN-1, num=LEN)
    f1 = interp1d(xold,dy_z_tac_data[:, i],kind='cubic')
    dy_z_tac_data_inter[:, i] = f1(xnew)
# for cut
head = 10
end = 10
dy_z_tac_data_inter = np.delete(dy_z_tac_data_inter, np.arange(head).tolist(), 0)
dy_z_tac_data_inter = np.delete(dy_z_tac_data_inter, (np.arange(end)*-1).tolist(), 0)
y_z_tac_data = np.delete(y_z_tac_data, np.arange(head).tolist(), 0)
y_z_tac_data = np.delete(y_z_tac_data, (np.arange(end)*-1).tolist(), 0)

for i in range(row):
    for j in range(col):
        index = (i * 2 + j) + 1
        yl = index * 3 - 1
        # print(index, yl)
        # print(all_tac_data.shape, np.arange(0, 1, 0.1), np.linspace(0,1,11))
        # ax = axs2[i][j].scatter(all_tac_data[:, yl-1],
        #                         all_tac_data[:, yl-0],
        #                         c=np.linspace(0, 1, num=all_tac_data.shape[0]),
        #                         cmap='spring', vmin=0, vmax=0.35)
        # fig2.colorbar(ax, orientation='horizontal')
        # # for num in range(all_tac_data.shape[0]):
        # #     axs2[i][j].plot(all_tac_data[num, yl],
        # #                     all_tac_data[num, yl-1], 'o', c=cm.OrRd(num/all_tac_data.shape[0]))
        axs2[i][j].plot(all_tac_data[:, yl-1] / all_tac_data[:, yl])
        axs2[i][j].plot(dy_z_tac_data_inter[:, index - 1])
        if _legend is False:
            axs2[i][j].legend()

            _legend = True
        # axs1[i][j].plot(pos_data, tac_data[:, i+j])
        # axs2[i][j].set_xlim(xmin, xmax)
        # axs2[i][j].set_ylim(ymin, ymax)
        axs2[i][j].set_ylabel(y_force[index-1])
        axs2[i][j].set_xlabel('time')

plt.show()