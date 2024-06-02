import matplotlib.pyplot as plt
import numpy as np

G2_fmeans = np.array([0.323, 0.184, 0.515, 0.524, 0.621, 0.315, 0.183, 0.486, 0.272, 0.119, 0.263, 0.095, 0.0, 0.389, 0.209, 0.326, 0.102, 0.239, 0.355, 0.112, 0.442, 0.539, 0.459, 0.605, 0.413, 0.264, 0.091, 0])
G2_fvar = np.array([0.00215, 0.00138, 0.075, 0.067, 0.084, 0.057, 0.00384, 0.012, 0.00751, 0.00103, 0.0064, 0.00019, 0, 0.00145, 0.0053, 0.0031, 0.00039, 0.0017, 0.0098, 0.00084, 0.057, 0.037, 0.0087, 0.049, 0.0078, 0.0031, 0.00107, 0])
G3_fmeans = np.array([0.349, 0.215, 0.556, 0.569, 0.665, 0.352, 0.211, 0.523, 0.315, 0.086, 0.286, 0.108, 0.478, 0.423, 0.219, 0.354, 0.136, 0.254, 0.386, 0.131, 0.484, 0.467, 0.524, 0.617, 0.465, 0.319, 0.097, 0])
G3_fvar = np.array([0.00477, 0.00205, 0.034, 0.031, 0.044, 0.009, 0.00751, 0.0092, 0.00415, 0.0034, 0.0099, 0.0012, 0.00511, 0.00238, 0.00974, 0.0102, 0.00111, 0.00123, 0.00541, 0.00203, 0.024, 0.010, 0.00941, 0.0208, 0.00818, 0.00214, 0.00541, 0])
G4_fmeans = np.array([0.359, 0.235, 0.596, 0.559, 0.635, 0.372, 0.231, 0.533, 0.355, 0.089, 0.316, 0.128, 0.548, 0.483, 0.239, 0.364, 0.196, 0.234, 0.396, 0.161, 0.524, 0.487, 0.544, 0.667, 0.475, 0.349, 0.107, 1.593])
G4_fvar = np.array([0.02477, 0.01305, 0.054, 0.071, 0.034, 0.029, 0.01751, 0.0192, 0.03415, 0.0154, 0.0199, 0.0112, 0.01511, 0.04238, 0.04974, 0.0112, 0.00111, 0.00123, 0.01541, 0.01203, 0.124, 0.050, 0.01941, 0.0208, 0.00818, 0.00214, 0.00541, 0.227])

G2_tmeans = np.array([1.67, 1.54, 1.75, 1.24, 5.73,
                      2.07, 3.01, 5.66, 2.07, 3.98,
                      1.27, 0.87, 0, 1.45, 2.09,
                      1.98, 1.67, 1.98, 1.45, 0.89,
                      3.77, 2.17, 1.53, 3.57, 2.27,
                      1.02, 0.78, 0])
G2_tvar = np.array([0.24353004, 0.19342678, 0.28188569, 0.21942362, 0.47723973,
                    0.23214946, 0.10392556, 0.20787959, 0.28160356, 0.37087905,
                    0.32779911, 0.28716744, 0, 0.19038136, 0.39917514,
                    0.13700095, 0.03997973, 0.43422337, 0.02247683, 0.09448951,
                    0.45549916, 0.25821922, 0.24495314, 0.11360982, 0.19931622,
                    0.02342368, 0.04635261, 0])
G3_tmeans = np.array([1.24, 1.12, 1.02, 0.98, 4.47,
                      1.13, 2.53, 4.17, 1.53, 3.55,
                      1.13, 0.85, 2.77, 1.21, 1.51,
                      1.75, 1.49, 1.76, 1.28, 0.77,
                      3.57, 1.96, 1.34, 3.18, 2.01,
                      0.97, 0.71, 0])
G3_tvar = np.array([0.46110063, 0.03876812, 0.25059384, 0.09562085, 0.02367365,
                    0.48772722, 0.2971253 , 0.29555709, 0.46475962, 0.04451384,
                    0.40824069, 0.23535711, 0.04753945, 0.23729059, 0.15741899,
                    0.19788794, 0.36982258, 0.17377336, 0.09528705, 0.31518153,
                    0.00479282, 0.25122235, 0.05473179, 0.11855848, 0.25901605,
                    0.30093036, 0.23711019, 0])
G4_tmeans = np.array([1.57, 1.23, 1.09, 1.38, 4.54,
                      1.27, 2.64, 4.34, 1.76, 5.70,
                      1.34, 0.95, 6.32, 1.73, 1.89,
                      1.80, 1.53, 2.09, 1.53, 1.14,
                      4.18, 2.31, 1.77, 4.86, 2.15,
                      1.07, 0.79, 5.77])
G4_tvar = np.array([0.30963238, 0.47091335, 0.38773048, 0.47015209, 0.04777455,
                    0.43721624, 0.22594313, 0.3928743 , 0.42187669, 0.32877809,
                    0.21835315, 0.49203849, 0.32562446, 0.4201814 , 0.18127838,
                    0.20214035, 0.08486589, 0.04265696, 0.01507135, 0.27213188,
                    0.2132708 , 0.43894241, 0.31547568, 0.31477057, 0.10993024,
                    0.26327785, 0.09777812, 0.39066961])

G1_sr = np.array([13, 0, 15, 3, 15, 0, 0, 15, 4, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 3, 15, 15, 7, 15, 14, 0]) / 15 * 100
G2_sr = np.array([9, 8, 13, 8, 9, 11, 13, 13, 12, 6, 9, 9, 0, 11, 11, 14, 10, 5, 9, 6, 10, 8, 14, 7, 12, 10, 13, 0]) / 15 * 100
G3_sr = np.array([13, 14, 14, 11, 14, 14, 14, 14, 14, 8, 13, 12, 9, 13, 14, 14, 13, 8, 12, 9, 11, 13, 14, 10, 13, 13, 14, 0]) / 15 * 100

linewidth = 2.0
fontsize=12
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : fontsize,
}

label = np.array([])
for i in range(G2_fvar.shape[0]):
    label = np.append(label, r'$id_{%d}$' % (i+1))
# label = (r'$id_{1}$', r'$id_{2}$', r'$id_{3}$')

fig_w, fig_h = 15.5, 7

colors = ['#D693BE', '#8EC8ED', '#F5B3A5', '#AED594']

fig, ax = plt.subplots(3, 1, figsize=(fig_w,fig_h))
bar_width = 0.25
var_width = 1.5
bar_x = np.arange(len(label))
r2 = [x + bar_width for x in bar_x]
r3 = [x + bar_width for x in r2]
ax[0].bar(x=bar_x, height=G1_sr, width=bar_width, color=colors[0],
                ecolor='#994487',
                capsize=var_width,
               )
ax[0].bar(x=r2, height=G2_sr, width=bar_width, color=colors[1],
                ecolor='#994487',
                capsize=var_width,
               )
SR1 = ax[0].bar(x=r3, height=G3_sr, width=bar_width, color=colors[2],
                ecolor='#994487',
                capsize=var_width,
               )
ax[0].set_xticks((r2+bar_x+r3)/3, labels=label, fontsize=fontsize)
ax[0].set_ylabel(r'$s_r (\%)$', font1)
ax[0].legend(('G1', 'G2', 'G3'), fontsize=fontsize, ncol=1, bbox_to_anchor=(1.05, 1))
ax[0].spines['top'].set_color('none')
ax[0].spines['right'].set_color('none')

ax[1].bar(x=bar_x, height=G2_fmeans, width=bar_width, color=colors[1],
               yerr=G2_fvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[1].bar(x=r2, height=G3_fmeans, width=bar_width, color=colors[2],
               yerr=G3_fvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[1].bar(x=r3, height=G4_fmeans, width=bar_width, color=colors[3],
               yerr=G4_fvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[1].set_xticks((r2+bar_x+r3)/3, labels=label, fontsize=fontsize)
ax[1].set_ylabel(r'$f_g (N)$', font1)
ax[1].legend(('G2', 'G3', 'G4'), fontsize=fontsize, ncol=1, bbox_to_anchor=(1.05, 1))
ax[1].spines['top'].set_color('none')
ax[1].spines['right'].set_color('none')

ax[2].bar(x=bar_x, height=G2_tmeans, width=bar_width, color=colors[1],
               yerr=G2_tvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[2].bar(x=r2, height=G3_tmeans, width=bar_width, color=colors[2],
               yerr=G3_tvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[2].bar(x=r3, height=G4_tmeans, width=bar_width, color=colors[3],
               yerr=G4_tvar,
               ecolor='#994487',
               capsize=var_width,
             )
ax[2].set_xticks((r2+bar_x+r3)/3, labels=label, fontsize=fontsize)
ax[2].set_ylabel(r'$t_g (s)$', font1)
# ax[2].set_xlabel(r'$t_g (s)$', font1)
ax[2].legend(('G2', 'G3', 'G4'), fontsize=fontsize, ncol=1, bbox_to_anchor=(1.05, 1))
ax[2].spines['top'].set_color('none')
ax[2].spines['right'].set_color('none')
plt.show()

