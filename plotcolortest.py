import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
N = 50
x = np.arange(0,1,0.0001)
y = np.arange(0,1,0.0001)
z = np.arange(0,10,0.001)

# 创建散点图
plt.scatter(x, y, s=100, c=z, cmap='Reds', vmin=0, vmax=1)

# 添加颜色条
plt.colorbar()

# 显示图形
plt.show()
