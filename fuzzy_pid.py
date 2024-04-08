import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# err and E_err as input, pid parameters as output
# kp, ki, kd have different regulations

# input
ferr_range = np.arange(-0.5, 0.6, 0.1, np.float32)  # 输入1
print('ferr_range:', ferr_range)
E_ferr_range = np.arange(-0.1, 0.11, 0.01, np.float32)    # 输入2
print('E_ferr_range:', E_ferr_range)
# output
pid_min = np.array([0.1, 0.001, 0.0001])
pid_max = np.array([10, 0.05, 0.05])
pid_mid = (pid_max + pid_min) / 2
kp_range = np.arange(pid_min[0], pid_max[0], 0.1, np.float32)
ki_range = np.arange(pid_min[1], pid_max[1], 0.0001, np.float32)
kd_range = np.arange(pid_min[2], pid_max[2], 0.0001, np.float32)

# create fuzzy control
ferr = ctrl.Antecedent(ferr_range, 'err')      # input1

E_ferr = ctrl.Antecedent(E_ferr_range, 'Eerr')            # input2

kp = ctrl.Consequent(kp_range, 'kp')   # output
ki = ctrl.Consequent(ki_range, 'ki')   # output
kd = ctrl.Consequent(kd_range, 'kd')   # output

# define class of fuzzy
# input1 and 3 levels
ferr['N'] = fuzz.trimf(ferr_range, [-0.5, -0.5, 0])
ferr['M'] = fuzz.trimf(ferr_range, [-0.5, 0, 0.5])
ferr['P'] = fuzz.trimf(ferr_range, [0, 0.5, 0.5])
# input2 and 3 levels
E_ferr['N'] = fuzz.trimf(E_ferr_range, [-0.1, -0.1, 0])
E_ferr['M'] = fuzz.trimf(E_ferr_range, [-0.1, 0, 0.1])
E_ferr['P'] = fuzz.trimf(E_ferr_range, [0, 0.1, 0.1])
# output kp and 3 levels
kp['N'] = fuzz.trimf(kp_range, [pid_min[0], pid_min[0], pid_mid[0]])
kp['M'] = fuzz.trimf(kp_range, [pid_min[0], pid_mid[0], pid_max[0]])
kp['P'] = fuzz.trimf(kp_range, [pid_mid[0], pid_max[0], pid_max[0]])
# output kp and 3 levels
ki['N'] = fuzz.trimf(kp_range, [pid_min[1], pid_min[1], pid_mid[1]])
ki['M'] = fuzz.trimf(kp_range, [pid_min[1], pid_mid[1], pid_max[1]])
ki['P'] = fuzz.trimf(kp_range, [pid_mid[1], pid_max[1], pid_max[1]])
# output kp and 3 levels
kd['N'] = fuzz.trimf(kp_range, [pid_min[2], pid_min[2], pid_mid[2]])
kd['M'] = fuzz.trimf(kp_range, [pid_min[2], pid_mid[2], pid_max[2]])
kd['P'] = fuzz.trimf(kp_range, [pid_mid[2], pid_max[2], pid_max[2]])

# define the defuzzify method
kp.defuzzify_method = 'centroid'
ki.defuzzify_method = 'centroid'
kd.defuzzify_method = 'centroid'



# N regulation
kp_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
                              (ferr['M'] & E_ferr['N'])),
                  consequent=kp['N'], label='rule N')

# 输出为M的规则
rule1 = ctrl.Rule(antecedent=((x_stain['P'] & x_oil['N']) |
                              (x_stain['N'] & x_oil['M']) |
                              (x_stain['M'] & x_oil['M']) |
                              (x_stain['P'] & x_oil['M']) |
                              (x_stain['N'] & x_oil['P'])),
                  consequent=y_powder['M'], label='rule M')

# 输出为P的规则
rule2 = ctrl.Rule(antecedent=((x_stain['M'] & x_oil['P']) |
                              (x_stain['P'] & x_oil['P'])),
                  consequent=y_powder['P'], label='rule P')

# 系统和运行环境初始化
system = ctrl.ControlSystem(rules=[rule0, rule1, rule2])
sim = ctrl.ControlSystemSimulation(system)

# 运行系统
sim.input['stain'] = 0.1
sim.input['oil'] = 0
sim.compute()
output_powder = sim.output['powder']

# 打印输出结果
print(output_powder)
