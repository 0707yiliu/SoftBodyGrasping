import numpy as np
import matplotlib.pyplot as plt

# collect data as list
datasets = [
    '20240315120107_cookedegg.npz',
    '20240315121000_chips.npz',
    '20240315120803_orange_small.npz',
    '20240315121331_tomato.npz',
    '20240315123237_bread.npz',
    '20240315123631_eggshell.npz',
    '20240315145611_bananawhole.npz',
    '20240315145744_banana.npz',
]
tac_datalists = [np.load(datasets[i])['loop_tac_data'] for i in range(len(datasets))]
pos_datalists = [np.load(datasets[i])['gripper_pos'] for i in range(len(datasets))]
for i in range(len(datasets)):

    if len(tac_datalists[i]) != len(pos_datalists[i]):
        pos_datalists[i] = np.delete(pos_datalists[i], 0, 0)
    print(len(tac_datalists[i]), len(pos_datalists[i]))


dataall = np.load('20240315150348_banana.npz')
print(dataall['loop_tac_data'].shape)
tac_data = dataall['loop_tac_data']
data_xlen = tac_data.shape[0]
data_xlen = np.linspace(0, data_xlen-1, data_xlen)
print(data_xlen, data_xlen.shape)
pos_data = dataall['gripper_pos']
# pos_data = np.delete(pos_data, 0, 0)
pos_data_len = pos_data.shape[0]
print(pos_data_len)
fig1 = plt.figure(1)
plt.plot(np.linspace(0, pos_data_len-1, pos_data_len), pos_data)
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
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        axs[i][j].plot(data_xlen, tac_data[:, i+j])
        axs[i][j].set_ylabel(ylabel[yl])
        axs[i][j].set_xlabel('time')

fig1, axs1 = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
for i in range(row):
    for j in range(col):
        if i == 1:
            yl = i * j + 6
        elif i == 0:
            yl = (i + 1) * j
        axs1[i][j].plot(pos_data, tac_data[:, i+j])
        axs1[i][j].set_ylabel(ylabel[yl])
        axs1[i][j].set_xlabel('finger pos')


plt.show()