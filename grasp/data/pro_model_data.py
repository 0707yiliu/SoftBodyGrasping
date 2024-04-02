import time

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


import os
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
data_path_dir = './model_data/'
dirs = os.listdir(data_path_dir)
file_item = 0
for file in dirs:
    tac_datalists = np.load(data_path_dir+file)['loop_tac_data']
    pos_datalists = np.load(data_path_dir+file)['gripper_pos']
    # print(file)
    file_item += 1
    print(file_item)
    if len(tac_datalists) != len(pos_datalists):
        pos_datalists = np.delete(pos_datalists, 0, 0)

#     # -------- clip -------------
#     max_index = np.abs(tac_datalists).argmax()
#     l_step = 100
#     r_step = 50
#     obj_class = 1 # -----------
#     # print(tac_datalists[(int(max_index / 12)), max_index % 12])
#
#     _maxmin_step = int(max_index / tac_datalists.shape[1])
#     start_step = _maxmin_step - l_step
#     end_step = _maxmin_step + r_step
#     clip_data = tac_datalists[start_step:end_step, :]
#     obj_class = np.ones(end_step - start_step) * obj_class
#     if file_item == 1:
#         labeldata = np.hstack((clip_data, obj_class.reshape(end_step - start_step, 1)))
#     else:
#         _labeldata = np.hstack((clip_data, obj_class.reshape(end_step - start_step, 1)))
#         labeldata = np.vstack((labeldata, _labeldata))
#   # print(end_step - start_step)
#   # print(labeldata.shape)
#   # time.sleep(10)
# np.savez('./clip_data/' + current_time +  '_clipped_data' + '.npz',
#              labeled_data=labeldata)
# print('done')
# ---------------------------

    # # ---------- plot ---------------------
    # _legend = False
    # # print(np.linspace(0, len(tac_datalists[0])-1, len(tac_datalists[0])))
    # # print(tac_datalists)
    # row = 2
    # col = 6
    # my_dpi=90
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # legends = ['bottle', 'chips', 'orange', 'tomato', 'bread', 'eggshell', 'wholebanana', 'banana']
    # ylabel = [
    #             'sensor1_x', 'sensor1_y', 'sensor1_z',
    #             'sensor2_x', 'sensor2_y', 'sensor2_z',
    #             'sensor3_x', 'sensor3_y', 'sensor3_z',
    #             'sensor4_x', 'sensor4_y', 'sensor4_z',
    # ]
    # fig, axs = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
    # for i in range(row):
    #     for j in range(col):
    #         if i == 1:
    #             yl = i * j + 6
    #         elif i == 0:
    #             yl = (i + 1) * j
    #
    #             # time_line = np.linspace(0, len(tac_datalists[k])-1, len(tac_datalists[k]))
    #             # tac_data[:,yl] = tac_datalists[:, yl]
    #             # # tac_data = FirstOrderLag(tac_data, 0.8)
    #             # axs[i][j].plot(time_line[tac_range_min:tac_range_max], tac_data[tac_range_min:tac_range_max, yl], color=colors[k], label=legends[k])
    #         axs[i][j].plot(np.linspace(0, len(tac_datalists)-1, len(tac_datalists)), tac_datalists[:, yl], color=colors[0], label=legends[0])
    #         if _legend is False:
    #             axs[i][j].legend()
    #             _legend = True
    #         axs[i][j].legend()
    #         # axs[i][j].plot(data_xlen, tac_data[:, i+j])
    #         axs[i][j].set_ylabel(ylabel[yl])
    #         axs[i][j].set_xlabel('time')
    # plt.show()
    # # --------------------------------------