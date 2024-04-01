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

tac_range_min = 74
tac_range_max = tac_range_min + 150
record = False
if record is False:
    tac_range_min = 1
    tac_range_max = -1
obj_class = 2
# collect data as list
datasets = [
    '20240315140451_kiwis.npz',
    # '20240315121000_chips.npz',
    # '20240315120803_orange_small.npz',
    # '20240315121331_tomato.npz',
    # '20240315123237_bread.npz', # useless
    # '20240315113504_eggshell.npz',
    # '20240315145611_bananawhole.npz',
    # '20240315150348_banana.npz',
    # '20240315140451_kiwis.npz',
    # '20240315133421_bottle.npz',
]
tac_datalists = [np.load(datasets[i])['loop_tac_data'] for i in range(len(datasets))]
pos_datalists = [np.load(datasets[i])['gripper_pos'] for i in range(len(datasets))]
for i in range(len(datasets)):
    if len(tac_datalists[i]) != len(pos_datalists[i]):
        pos_datalists[i] = np.delete(pos_datalists[i], 0, 0)
    print(len(tac_datalists[i]), len(pos_datalists[i]))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
legends = ['bottle', 'chips', 'orange', 'tomato', 'bread', 'eggshell', 'wholebanana', 'banana']

dataall = np.load(datasets[0])
print(dataall['loop_tac_data'].shape)
tac_data = dataall['loop_tac_data']
data_xlen = tac_data.shape[0]
data_xlen = np.linspace(0, data_xlen-1, data_xlen)
print(data_xlen, data_xlen.shape)
pos_data = dataall['gripper_pos']
# pos_data = np.delete(pos_data, 0, 0)
pos_data_len = pos_data.shape[0]
print(pos_data_len)
# # figure
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

for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        for k in range(len(datasets)):
            time_line = np.linspace(0, len(tac_datalists[k])-1, len(tac_datalists[k]))
            tac_data[:,yl] = tac_datalists[k][:, yl]
            # tac_data = FirstOrderLag(tac_data, 0.8)
            axs[i][j].plot(time_line[tac_range_min:tac_range_max], tac_data[tac_range_min:tac_range_max, yl], color=colors[k], label=legends[k])
            # axs[i][j].plot(np.linspace(0, len(tac_datalists[k])-1, len(tac_datalists[k])), tac_datalists[k][:, yl], color=colors[k], label=legends[k])
        if _legend is False:
            axs[i][j].legend()
            _legend = True
        axs[i][j].legend()
        # axs[i][j].plot(data_xlen, tac_data[:, i+j])
        axs[i][j].set_ylabel(ylabel[yl])
        axs[i][j].set_xlabel('time')

# fig1, axs1 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
# _legend = False
# for i in range(row):
#     for j in range(col):
#         if i == 1:
#             yl = i * j + 6
#         elif i == 0:
#             yl = (i + 1) * j
#         for k in range(len(datasets)):
#             axs1[i][j].plot(pos_datalists[k], tac_datalists[k][:, yl], color=colors[k], label=legends[k])
#         if _legend is False:
#             axs1[i][j].legend()
#             _legend = True
#         # axs1[i][j].plot(pos_data, tac_data[:, i+j])
#         axs1[i][j].set_ylabel(ylabel[yl])
#         axs1[i][j].set_xlabel('finger pos')

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

if record is True:
    print(datasets[0][:-4]+'_labeled')
    labeldata = tac_data[tac_range_min:tac_range_max, :].transpose()
    print("origin label data", labeldata.shape)
    grasp_step = 100 - 12


    obj_class = np.ones(labeldata.shape[1]) * obj_class
    labeldata = np.vstack((labeldata, obj_class))
    print("obj label data", labeldata.shape)

    en_grasp = np.ones(labeldata.shape[1])
    en_grasp[:grasp_step] = en_grasp[:grasp_step] * -1
    labeldata = np.vstack((labeldata, en_grasp))
    print("enable grasp label data", labeldata.shape)

    np.savez('./labeled_data/' + datasets[0][:-4]+'_labeled' + '.npz',
                                     loop_tac_data=tac_data[tac_range_min:tac_range_max, :],
                                     labeled_data=labeldata)


