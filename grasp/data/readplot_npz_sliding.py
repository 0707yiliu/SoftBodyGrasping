import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs

# collect data as list
datasets = [
    # '20240315120107_cookedegg.npz',
    # '20240315121000_chips.npz',
    # '20240315120803_orange_small.npz',
    '20240411181401_1force_cup_lift1-0.005-0.005.npz',
    # '20240410140411_2force_cup_1-0.005-0.005.npz',
    # '20240315123237_bread.npz', # useless
    # '20240315113504_eggshell.npz',
    # '20240315145611_bananawhole.npz',
    # '20240315145744_banana.npz',
    # '20240315140451_kiwis.npz',
    # '20240315133421_bottle.npz',
]
tac_datalists = [np.load(datasets[i])['loop_tac_data'] for i in range(len(datasets))]
pos_datalists = [np.load(datasets[i])['gripper_pos'] for i in range(len(datasets))]
_tac_datalists = [np.load(datasets[i])['_tac_data'] for i in range(len(datasets))]
print(tac_datalists[0].shape, _tac_datalists[0].shape)
all_tac_data = np.vstack((tac_datalists[0], _tac_datalists[0]))
d_all_tac_data = FirstOrderLag(all_tac_data, 0.5)
all_tac_data = FirstOrderLag(all_tac_data, 0.5)
hz_time = 0.5
hz = 1 / hz_time
print('---', all_tac_data.shape)
d_all_tac_data = np.vstack((np.zeros(all_tac_data.shape[1]), d_all_tac_data))
for i in range(d_all_tac_data.shape[0]-1):
    d_all_tac_data[i, :] = (d_all_tac_data[i+1, :] - d_all_tac_data[i,:]) / hz_time
d_all_tac_data = np.delete(d_all_tac_data, -1, 0)

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
stay_item = 30
# print(tac_datalists[0].shape)
# print(_tac_datalists[0].shape)
fig.suptitle(datasets[0], fontsize=20)
legends = ['squeeze', 'hold', 'lift']
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        for k in range(len(datasets)):
            axs[i][j].plot(np.linspace(0,
                                       len(tac_datalists[k])-1,
                                       len(tac_datalists[k])),
                           tac_datalists[k][:, yl],
                           color=colors[k], label=legends[k])

            axs[i][j].plot(np.linspace(len(tac_datalists[k]),
                                       len(tac_datalists[k]) + stay_item - 1,
                                       stay_item),
                           _tac_datalists[k][:stay_item, yl],
                           color=colors[k + 1], label=legends[k + 1])
            axs[i][j].plot(np.linspace(len(tac_datalists[k]) + stay_item,
                                       len(tac_datalists[k]) + len(_tac_datalists[k]) - 1,
                                       len(_tac_datalists[k]) - stay_item),
                           _tac_datalists[k][stay_item:, yl],
                           color=colors[k + 2], label=legends[k + 2])

            # print(len(tac_datalists[k]), len(_tac_datalists[k]))
            # axs[i][j].plot(np.linspace(len(tac_datalists[k]),
            #                            len(tac_datalists[k]) + len(_tac_datalists[k]) - 1,
            #                            len(_tac_datalists[k])),
            #                _tac_datalists[k][:, yl],
            #                color=colors[k + 3], label=legends[k + 2])
        if _legend is False:
            axs[i][j].legend()
            _legend = True
        axs[i][j].legend()
        # axs[i][j].plot(data_xlen, tac_data[:, i+j])
        axs[i][j].set_ylabel(ylabel[yl])
        axs[i][j].set_xlabel('time')

fig1, axs1 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
_legend = False
legends = ['deriv', 'origin']
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        axs1[i][j].plot(np.linspace(0,
                                    d_all_tac_data.shape[0]-1,
                                    d_all_tac_data.shape[0]),
                        d_all_tac_data[:, yl],
                        color=colors[0], label=legends[k])
        axs1[i][j].plot(np.linspace(0,
                                    all_tac_data.shape[0] - 1,
                                    all_tac_data.shape[0]),
                        all_tac_data[:, yl],
                        color=colors[1], label=legends[k + 1])

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
y_force = ['sensro1_Y', 'sensro2_Y','sensro3_Y','sensro4_Y',]
fig2, axs2 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
_legend = False
# for i in range(row):
#     for j in range(col):
#         index = (i * 2 + j) + 1
#         yl = index * 3 - 1
#         print(index, yl)
#         x = all_tac_data[:, yl]
#         y = all_tac_data[:, yl-1]
#         ps = np.stack((x, y), axis=1)
#         segments = np.stack((ps[:-1], ps[1:]), axis=1)
#         cmap = 'jet'  # jet, hsv等也是常用的颜色映射方案
#         colors = color_map(x[:-1], cmap)
#         colors = color_map(y[:-1], cmap)
#         line_segments = LineCollection(segments, colors=colors, linewidths=3, linestyles='solid', cmap=cmap)
#         # fig, ax = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
#         axs2[i][j].set_xlim(np.min(x) - 0.1, np.max(x) + 0.1)
#         axs2[i][j].set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)
#         axs2[i][j].add_collection(line_segments)
#         cb = fig2.colorbar(line_segments, cmap='jet')
#         if _legend is False:
#             axs2[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs2[i][j].set_ylabel(y_force[index-1])
#         axs2[i][j].set_xlabel('Z')


for i in range(row):
    for j in range(col):
        index = (i * 2 + j) + 1
        yl = index * 3 - 1
        # print(index, yl)
        ax = axs2[i][j].scatter(all_tac_data[:, yl-1],
                                all_tac_data[:, yl],
                                c=np.arange(0, 1, 1/all_tac_data.shape[0]),
                                cmap='cool')
        fig2.colorbar(ax, orientation='horizontal')
        # for num in range(all_tac_data.shape[0]):
        #     axs2[i][j].plot(all_tac_data[num, yl],
        #                     all_tac_data[num, yl-1], 'o', c=cm.OrRd(num/all_tac_data.shape[0]))
        axs2[i][j].plot(all_tac_data[:, yl-1],
                        all_tac_data[:, yl])
        # cmap = cm.OrRd
        # norm = mcolors.Normalize(vmin=0, vmax=100)
        # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
        #              orientation='horizontal', label='spring')

        # cmap1 = copy.copy(cm.OrRd)
        # norm1 = mcolors.Normalize(vmin=0, vmax=100)
        # im1 = cm.ScalarMappable(norm=norm1, cmap=cmap1)
        # cbar1 = fig2.colorbar(
        #     im1, cax=axs2[i][j], orientation='horizontal',
        #     ticks=np.linspace(0, 100, 11),
        #     label='colorbar with Normalize'
        # )
        # axs1[i][j].plot(np.linspace(0,
        #                             all_tac_data.shape[0] - 1,
        #                             all_tac_data.shape[0]),
        #                 all_tac_data[:, yl],
        #                 color=colors[1])

        if _legend is False:
            axs2[i][j].legend()
            _legend = True
        # axs1[i][j].plot(pos_data, tac_data[:, i+j])
        axs2[i][j].set_ylabel('Z')
        axs2[i][j].set_xlabel(y_force[index-1])

# fig1, axs1 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
# _legend = False
# for i in range(row):
#     for j in range(col):
#         if i == 1:
#             yl = i * j + 6
#         elif i == 0:
#             yl = (i + 1) * j
#         for k in range(len(datasets)):
#             d_pos_datalists = np.zeros(len(pos_datalists[k]) - 1)
#             d_tac_datalists = np.zeros(len(tac_datalists[k][:, yl]) - 1)
#             timeline = np.linspace(0, len(tac_datalists[k]) - 1, len(tac_datalists[k]))
#             for data_len in range(len(pos_datalists[k]) - 1):
#                 # print(data_len)
#                 d_pos_datalists[data_len] = pos_datalists[k][data_len + 1] -  pos_datalists[k][data_len]
#                 d_tac_datalists[data_len] = (tac_datalists[k][data_len + 1, yl] - tac_datalists[k][data_len, yl]) * 1000
#             # axs1[i][j].plot(d_pos_datalists, d_tac_datalists, color=colors[k], label=legends[k])
#             b, a = signal.butter(8, 0.2, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
#             # d_pos_datalists = signal.filtfilt(b, a, d_pos_datalists)
#             # d_tac_datalists = signal.filtfilt(b, a, d_tac_datalists)
#             # print(len(d_pos_datalists), len(d_tac_datalists))
#             d_pos_datalists = FirstOrderLag(d_pos_datalists, 0.8)
#             # d_tac_datalists = ArithmeticAverage(d_tac_datalists, 30)
#             axs1[i][j].plot(np.linspace(0, len(d_pos_datalists) - 1, len(d_pos_datalists)),
#                             d_pos_datalists,
#                             color=colors[k+1], label=legends[k])
#             axs1[i][j].plot(np.linspace(0, len(d_tac_datalists) - 1, len(d_tac_datalists)),
#                             d_tac_datalists,
#                             color=colors[k], label=legends[k])
#
#         if _legend is False:
#             axs1[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs1[i][j].set_ylabel(ylabel[yl])
#         axs1[i][j].set_xlabel('finger pos')







plt.show()
