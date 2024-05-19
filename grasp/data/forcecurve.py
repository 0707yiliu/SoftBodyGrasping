import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

linewidth = 2.0
fontsize=16
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : fontsize,
}
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.text(x=1.1,  # 文本x轴坐标
         y=0.63,  # 文本y轴坐标
         s='broken boundary',  # 文本内容
         ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=14, color='black',
                       family='Arial',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                       weight='medium',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                       )  # 字体属性设置
         )

plt.text(x=1.1,  # 文本x轴坐标
         y=0.44,  # 文本y轴坐标
         s='slip boundary',  # 文本内容
         ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=14, color='black',
                       family='Arial',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                       weight='medium',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                       )  # 字体属性设置
         )

len = 100
kill = 5
extend_x = np.linspace(len, len+len/5, num=len)
extend_rigid=np.ones(extend_x.shape[0])
extend_soft=np.ones(extend_x.shape[0])
extend_fragile=np.zeros(extend_x.shape[0])
x = np.linspace(0,len,num=len)
extend_xx = np.hstack((x, extend_x))
rigid = np.linspace(0, 1, num=len)
rigid = np.hstack((rigid, extend_rigid))
# rigid[-kill:] = 1
plt.plot(extend_xx, rigid, linewidth=linewidth,label=r"rigid objects")
soft = x ** 3
soft=soft/soft.max()
soft = np.hstack((soft, extend_soft))
# soft[-kill:] = 1
plt.plot(extend_xx, soft, linewidth=linewidth,label=r"deformable objects")
fragile = x ** 8
fragile=fragile/fragile.max()
fragile[-kill:] = 0
fragile = np.hstack((fragile, extend_fragile))
plt.plot(extend_xx, fragile, linewidth=linewidth,label=r"fragile objects")

slip_line = np.ones(extend_xx.shape[0]) * 0.425
plt.plot(extend_xx, slip_line, linestyle='--', color='chartreuse', linewidth=linewidth)
break_line = np.ones(extend_xx.shape[0]) * 0.615
plt.plot(extend_xx, break_line, linestyle='--', color='r', linewidth=linewidth)

plt.xlabel('time step', font1)
plt.ylabel('Force', font1)
plt.xlim(0,120)
plt.legend(fontsize=fontsize-2)


plt.show()