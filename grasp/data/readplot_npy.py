import numpy as np
import matplotlib.pyplot as plt


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
fig1 = plt.figure()
plt.plot(np.linspace(0, pos_data_len-1, pos_data_len), pos_data)
my_dpi=96
row = 2
col = 6
fig, axs = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
for i in range(row):
    for j in range(col):
        axs[i][j].plot(data_xlen, tac_data[:, i+j])
plt.show()