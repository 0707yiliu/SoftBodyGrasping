import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# err and E_err as input, pid parameters as output
# kp, ki, kd have different regulations

# input
ferr_range = np.linspace(-0.4, 0.4, num=50)  # 输入1
print('ferr_range:', ferr_range)
E_ferr_range = np.linspace(-0.4, 0.4, num=50)   # 输入2
print('E_ferr_range:', E_ferr_range)
# output
pid_min = np.array([0.1, 0.01, 0.01])
pid_max = np.array([10, 5, 2])
pid_mid = (pid_max + pid_min) / 2
kp_range = np.linspace(pid_min[0], pid_max[0], num=50)
ki_range = np.linspace(pid_min[1], pid_max[1], num=50)
kd_range = np.linspace(pid_min[2], pid_max[2], num=50)

# create fuzzy control
ferr = ctrl.Antecedent(ferr_range, 'ferr')      # input1

E_ferr = ctrl.Antecedent(E_ferr_range, 'fEerr')            # input2

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
print(len(kp_range), len(ki_range), len(kd_range))
kp['N'] = fuzz.trimf(kp_range, [pid_min[0], pid_min[0], pid_mid[0]])
kp['M'] = fuzz.trimf(kp_range, [pid_min[0], pid_mid[0], pid_max[0]])
kp['P'] = fuzz.trimf(kp_range, [pid_mid[0], pid_max[0], pid_max[0]])
# # output kp and 3 levels
# ki['N'] = fuzz.trimf(kp_range, [pid_min[1], pid_min[1], pid_mid[1]])
# ki['M'] = fuzz.trimf(kp_range, [pid_min[1], pid_mid[1], pid_max[1]])
# ki['P'] = fuzz.trimf(kp_range, [pid_mid[1], pid_max[1], pid_max[1]])
# # output kp and 3 levels
# kd['N'] = fuzz.trimf(kp_range, [pid_min[2], pid_min[2], pid_mid[2]])
# kd['M'] = fuzz.trimf(kp_range, [pid_min[2], pid_mid[2], pid_max[2]])
# kd['P'] = fuzz.trimf(kp_range, [pid_mid[2], pid_max[2], pid_max[2]])

# define the defuzzify method
kp.defuzzify_method = 'centroid'
# ki.defuzzify_method = 'centroid'
# kd.defuzzify_method = 'centroid'


# -----------------------Kp----------------------------------------
# N regulation
kp_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
                              (ferr['M'] & E_ferr['N']) |
                              (ferr['M'] & E_ferr['M'])),
                  consequent=kp['N'], label='rule N')
# M regulation
kp_rule1 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['N']) |
                              (ferr['N'] & E_ferr['M']) |
                              (ferr['P'] & E_ferr['M'])),
                  consequent=kp['M'], label='rule M')
# P regulation
kp_rule2 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['P']) |
                              (ferr['M'] & E_ferr['P']) |
                              (ferr['N'] & E_ferr['P']) ),
                  consequent=kp['P'], label='rule P')
# # ------------------------Ki---------------------------------------
# # N regulation
# ki_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
#                               (ferr['N'] & E_ferr['M']) |
#                               (ferr['N'] & E_ferr['P']) |
#                               (ferr['P'] & E_ferr['N']) |
#                               (ferr['P'] & E_ferr['P']) ),
#                   consequent=ki['N'], label='rule N')
# # M regulation
# ki_rule1 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['N']) |
#                               (ferr['M'] & E_ferr['P']) ),
#                   consequent=ki['M'], label='rule M')
# # P regulation
# ki_rule2 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['M']) |
#                                 (ferr['M'] & E_ferr['M'])),
#                   consequent=ki['P'], label='rule P')
# # -------------------------Kd--------------------------------------
# # N regulation
# kd_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['M']) |
#                               (ferr['M'] & E_ferr['M']) |
#                               (ferr['P'] & E_ferr['M']) ),
#                   consequent=kd['N'], label='rule N')
# # M regulation
# kd_rule1 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['N']) |
#                               (ferr['M'] & E_ferr['P']) ),
#                   consequent=kd['M'], label='rule M')
# # P regulation
# kd_rule2 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
#                               (ferr['N'] & E_ferr['P']) |
#                               (ferr['P'] & E_ferr['P']) |
#                               (ferr['P'] & E_ferr['N']) ),
#                   consequent=kd['P'], label='rule P')
# # ---------------------------------------------------------------
# init fuzzy system
ad_kp = ctrl.ControlSystem(rules=[kp_rule0, kp_rule1, kp_rule2])
_kp = ctrl.ControlSystemSimulation(ad_kp)
# ad_ki = ctrl.ControlSystem(rules=[ki_rule0, ki_rule1, ki_rule2])
# _ki = ctrl.ControlSystemSimulation(ad_ki)
# ad_kd = ctrl.ControlSystem(rules=[kd_rule0, kd_rule1, kd_rule2])
# _kd = ctrl.ControlSystemSimulation(ad_kd)
_ferr = -0.2
_fEerr = 0.1
# run fuzzy system
_kp.input['ferr'] = _ferr
_kp.input['fEerr'] = _fEerr
_kp.compute()
kp_out = _kp.output['kp']

# _ki.input['ferr'] = _ferr
# _ki.input['fEerr'] = _fEerr
# _ki.compute()
# ki_out = _ki.output['ki']
#
# _kd.input['ferr'] = _ferr
# _kd.input['fEerr'] = _fEerr
# _kd.compute()
# kd_out = _kd.output['kd']

#
print(kp_out)
