import numpy as np
import matplotlib.pyplot as plt


dataall = np.load('20240313154408_banana.npz')
print(dataall['loop_tac_data'].shape)
data = dataall['loop_tac_data']
data_xlen = data.shape[0]
data_xlen = np.linspace(0, data_xlen-1, data_xlen)
print(data_xlen, data_xlen.shape)
my_dpi=96
row = 2
col = 6
fig, axs = plt.subplots(row, col, figsize=(480/my_dpi,480/my_dpi),dpi=my_dpi, sharex=False, sharey=False)
for i in range(row):
    for j in range(col):
        axs[i][j].plot(data_xlen, data[:, i+j])
plt.show()