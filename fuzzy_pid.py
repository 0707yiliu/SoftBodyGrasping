import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

class Fuzzy_PID:
    def __init__(self,
              err_min=-2, err_max=2, E_err_min=-1, E_err_max=1,
              kp_min=1, kp_max=3,
              ki_min=0.01, ki_max=0.5,
              kd_min=0.01, kd_max=0.1):

        fuzzy_num = 50
        # input
        ferr_min = err_min
        ferr_max = err_max
        ferr_mid = (ferr_max + ferr_max) / 2
        ferr_range = np.linspace(ferr_min, ferr_max, num=fuzzy_num)  # input 1

        fEerr_min = E_err_min
        fEerr_max = E_err_max
        fEerr_mid = (fEerr_max + fEerr_max) / 2
        E_ferr_range = np.linspace(fEerr_min, fEerr_max, num=fuzzy_num)  # input 2

        self.k_i_d_ratio = 100
        pid_min = np.array([kp_min, ki_min * self.k_i_d_ratio, kd_min * self.k_i_d_ratio])
        pid_max = np.array([kp_max, ki_max * self.k_i_d_ratio, kd_max * self.k_i_d_ratio])
        pid_mid = (pid_max + pid_min) / 2
        kp_range = np.linspace(pid_min[0], pid_max[0], num=fuzzy_num)
        ki_range = np.linspace(pid_min[1], pid_max[1], num=fuzzy_num)
        kd_range = np.linspace(pid_min[2], pid_max[2], num=fuzzy_num)

        # create fuzzy control
        ferr = ctrl.Antecedent(ferr_range, 'ferr')  # input1

        E_ferr = ctrl.Antecedent(E_ferr_range, 'fEerr')  # input2

        kp = ctrl.Consequent(kp_range, 'kp')  # output
        ki = ctrl.Consequent(ki_range, 'ki')  # output
        kd = ctrl.Consequent(kd_range, 'kd')  # output

        # define class of fuzzy
        # input1 and 3 levels
        ferr['N'] = fuzz.trimf(ferr_range, [ferr_min, ferr_min, ferr_mid])
        ferr['M'] = fuzz.trimf(ferr_range, [ferr_min, ferr_mid, ferr_max])
        ferr['P'] = fuzz.trimf(ferr_range, [ferr_mid, ferr_max, ferr_max])
        # input2 and 3 levels
        E_ferr['N'] = fuzz.trimf(E_ferr_range, [fEerr_min, fEerr_min, fEerr_mid])
        E_ferr['M'] = fuzz.trimf(E_ferr_range, [fEerr_min, fEerr_mid, fEerr_max])
        E_ferr['P'] = fuzz.trimf(E_ferr_range, [fEerr_mid, fEerr_max, fEerr_max])
        # output kp and 3 levels
        print(len(kp_range), len(ki_range), len(kd_range))
        kp['N'] = fuzz.trimf(kp_range, [pid_min[0], pid_min[0], pid_mid[0]])
        kp['M'] = fuzz.trimf(kp_range, [pid_min[0], pid_mid[0], pid_max[0]])
        kp['P'] = fuzz.trimf(kp_range, [pid_mid[0], pid_max[0], pid_max[0]])
        # # output kp and 3 levels
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

        # -----------------------Kp----------------------------------------
        # N regulation
        kp_rule0 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['N']) |
                                         (ferr['N'] & E_ferr['P']) |
                                         (ferr['P'] & E_ferr['P']) |
                                         (ferr['N'] & E_ferr['N'])),
                             consequent=kp['N'], label='rule N')
        # M regulation
        kp_rule1 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['N']) |
                                         (ferr['N'] & E_ferr['M']) |
                                         (ferr['P'] & E_ferr['M']) |
                                         (ferr['M'] & E_ferr['P'])),
                             consequent=kp['M'], label='rule M')
        # P regulation
        kp_rule2 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['M']) ),
                             consequent=kp['P'], label='rule P')
        # # ------------------------Ki---------------------------------------
        # N regulation
        ki_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
                                         (ferr['P'] & E_ferr['P'])),
                             consequent=ki['N'], label='rule N')
        # M regulation
        ki_rule1 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['N']) |
                                         (ferr['N'] & E_ferr['P'])),
                             consequent=ki['M'], label='rule M')
        # P regulation
        ki_rule2 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['M']) |
                                         (ferr['M'] & E_ferr['M']) |
                                         (ferr['M'] & E_ferr['P']) |
                                         (ferr['M'] & E_ferr['N']) |
                                         (ferr['N'] & E_ferr['M'])),
                             consequent=ki['P'], label='rule P')
        # # -------------------------Kd--------------------------------------
        # N regulation
        kd_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['M']) |
                                         (ferr['M'] & E_ferr['M']) |
                                         (ferr['P'] & E_ferr['M'])),
                             consequent=kd['N'], label='rule N')
        # M regulation
        kd_rule1 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['N']) |
                                         (ferr['M'] & E_ferr['P'])),
                             consequent=kd['M'], label='rule M')
        # P regulation
        kd_rule2 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
                                         (ferr['N'] & E_ferr['P']) |
                                         (ferr['P'] & E_ferr['P']) |
                                         (ferr['P'] & E_ferr['N'])),
                             consequent=kd['P'], label='rule P')
        # # ---------------------------------------------------------------

        # init fuzzy system
        ad_kp = ctrl.ControlSystem(rules=[kp_rule0, kp_rule1, kp_rule2])
        self._kp = ctrl.ControlSystemSimulation(ad_kp)
        ad_ki = ctrl.ControlSystem(rules=[ki_rule0, ki_rule1, ki_rule2])
        self._ki = ctrl.ControlSystemSimulation(ad_ki)
        ad_kd = ctrl.ControlSystem(rules=[kd_rule0, kd_rule1, kd_rule2])
        self._kd = ctrl.ControlSystemSimulation(ad_kd)

    def compute(self, err, E_err):
        # run fuzzy system
        self._kp.input['ferr'] = err
        self._kp.input['fEerr'] = E_err
        self._kp.compute()
        kp_out = self._kp.output['kp']

        self._ki.input['ferr'] = err
        self._ki.input['fEerr'] = E_err
        self._ki.compute()
        ki_out = self._ki.output['ki']

        self._kd.input['ferr'] = err
        self._kd.input['fEerr'] = E_err
        self._kd.compute()
        kd_out = self._kd.output['kd']

        return kp_out, ki_out / self.k_i_d_ratio, kd_out / self.k_i_d_ratio

myfuzz_pid = Fuzzy_PID()
e = -0.5063523042409486
ee = -0.4935227632522583

p,i,d = myfuzz_pid.compute(e, ee)
print(
(p*e + i*e + d * ee) * 0.01, p,i,d

)
e = -0.01063523042409486
ee = -0.004935227632522583
p,i,d = myfuzz_pid.compute(e, ee)
print(
(p*e + i*e + d * ee) * 0.01, p,i,d

)
# -0.21282954098869034 -0.21282954098869034
# -0.7063523042409486 -0.4935227632522583
# -0.9354917514531679 -0.22913944721221924
# -0.8513969886510439 0.08409476280212402
# -0.35564930321860977 0.4957476854324341
# -0.11556757571387954 0.24008172750473022
# 0.04685120699714951 0.16241878271102905
# 0.12056737063240341 0.0737161636352539
# 0.1919949760892351 0.07142760545683169
# 0.22307108757805158 0.031076111488816477
# 0.2495901030094557 0.026519015431404114
# 0.26657914040397934 0.01698903739452365
# 0.2971912142426901 0.030612073838710785
# 0.31119553414892487 0.014004319906234741
# 0.3110985350401335 -9.699910879135132e-05
# 0.3096959855945044 -0.0014025494456291199

# # err and E_err as input, pid parameters as output
# # kp, ki, kd have different regulations
#
# # input
# ferr_min = -0.4
# ferr_max = 0.4
# ferr_mid = (ferr_max + ferr_max) / 2
# ferr_range = np.linspace(ferr_min, ferr_max, num=50)  # 输入1
# print('ferr_range:', ferr_range)
# fEerr_min = -0.3
# fEerr_max = 0.3
# fEerr_mid = (fEerr_max + fEerr_max) / 2
# E_ferr_range = np.linspace(fEerr_min, fEerr_max, num=50)   # 输入2
# print('E_ferr_range:', E_ferr_range)
# # output
# pid_min = np.array([0.1, 0.01, 0.01])
# pid_max = np.array([10, 5, 2])
# pid_mid = (pid_max + pid_min) / 2
# kp_range = np.linspace(pid_min[0], pid_max[0], num=50)
# ki_range = np.linspace(pid_min[1], pid_max[1], num=50)
# kd_range = np.linspace(pid_min[2], pid_max[2], num=50)
#
# # create fuzzy control
# ferr = ctrl.Antecedent(ferr_range, 'ferr')      # input1
#
# E_ferr = ctrl.Antecedent(E_ferr_range, 'fEerr')            # input2
#
# kp = ctrl.Consequent(kp_range, 'kp')   # output
# ki = ctrl.Consequent(ki_range, 'ki')   # output
# kd = ctrl.Consequent(kd_range, 'kd')   # output
#
# # define class of fuzzy
# # input1 and 3 levels
# ferr['N'] = fuzz.trimf(ferr_range, [ferr_min, ferr_min, ferr_mid])
# ferr['M'] = fuzz.trimf(ferr_range, [ferr_min, ferr_mid, ferr_max])
# ferr['P'] = fuzz.trimf(ferr_range, [ferr_mid, ferr_max, ferr_max])
# # input2 and 3 levels
# E_ferr['N'] = fuzz.trimf(E_ferr_range, [fEerr_min, fEerr_min, fEerr_mid])
# E_ferr['M'] = fuzz.trimf(E_ferr_range, [fEerr_min, fEerr_mid, fEerr_max])
# E_ferr['P'] = fuzz.trimf(E_ferr_range, [fEerr_mid, fEerr_max, fEerr_max])
# # output kp and 3 levels
# print(len(kp_range), len(ki_range), len(kd_range))
# kp['N'] = fuzz.trimf(kp_range, [pid_min[0], pid_min[0], pid_mid[0]])
# kp['M'] = fuzz.trimf(kp_range, [pid_min[0], pid_mid[0], pid_max[0]])
# kp['P'] = fuzz.trimf(kp_range, [pid_mid[0], pid_max[0], pid_max[0]])
# # # output kp and 3 levels
# ki['N'] = fuzz.trimf(kp_range, [pid_min[1], pid_min[1], pid_mid[1]])
# ki['M'] = fuzz.trimf(kp_range, [pid_min[1], pid_mid[1], pid_max[1]])
# ki['P'] = fuzz.trimf(kp_range, [pid_mid[1], pid_max[1], pid_max[1]])
# # output kp and 3 levels
# kd['N'] = fuzz.trimf(kp_range, [pid_min[2], pid_min[2], pid_mid[2]])
# kd['M'] = fuzz.trimf(kp_range, [pid_min[2], pid_mid[2], pid_max[2]])
# kd['P'] = fuzz.trimf(kp_range, [pid_mid[2], pid_max[2], pid_max[2]])
#
# # define the defuzzify method
# kp.defuzzify_method = 'centroid'
# ki.defuzzify_method = 'centroid'
# kd.defuzzify_method = 'centroid'
#
#
# # -----------------------Kp----------------------------------------
# # N regulation
# kp_rule0 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['N']) |
#                               (ferr['N'] & E_ferr['P']) |
#                               (ferr['M'] & E_ferr['M'])),
#                   consequent=kp['N'], label='rule N')
# # M regulation
# kp_rule1 = ctrl.Rule(antecedent=((ferr['M'] & E_ferr['N']) |
#                               (ferr['N'] & E_ferr['M']) |
#                               (ferr['P'] & E_ferr['M']) |
#                               (ferr['M'] & E_ferr['P'])),
#                   consequent=kp['M'], label='rule M')
# # P regulation
# kp_rule2 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['P']) |
#                               (ferr['N'] & E_ferr['N'])),
#                   consequent=kp['P'], label='rule P')
# # # ------------------------Ki---------------------------------------
# # N regulation
# ki_rule0 = ctrl.Rule(antecedent=((ferr['N'] & E_ferr['N']) |
#                               (ferr['P'] & E_ferr['P']) ),
#                   consequent=ki['N'], label='rule N')
# # M regulation
# ki_rule1 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['N']) |
#                               (ferr['N'] & E_ferr['P']) ),
#                   consequent=ki['M'], label='rule M')
# # P regulation
# ki_rule2 = ctrl.Rule(antecedent=((ferr['P'] & E_ferr['M']) |
#                                 (ferr['M'] & E_ferr['M']) |
#                                 (ferr['M'] & E_ferr['P']) |
#                                 (ferr['M'] & E_ferr['N']) |
#                                 (ferr['N'] & E_ferr['M'])),
#                   consequent=ki['P'], label='rule P')
# # # -------------------------Kd--------------------------------------
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
# # # ---------------------------------------------------------------
# # init fuzzy system
# ad_kp = ctrl.ControlSystem(rules=[kp_rule0, kp_rule1, kp_rule2])
# _kp = ctrl.ControlSystemSimulation(ad_kp)
# ad_ki = ctrl.ControlSystem(rules=[ki_rule0, ki_rule1, ki_rule2])
# _ki = ctrl.ControlSystemSimulation(ad_ki)
# ad_kd = ctrl.ControlSystem(rules=[kd_rule0, kd_rule1, kd_rule2])
# _kd = ctrl.ControlSystemSimulation(ad_kd)
# _ferr = -0.1
# _fEerr = -0.001
# # run fuzzy system
# _kp.input['ferr'] = _ferr
# _kp.input['fEerr'] = _fEerr
# _kp.compute()
# kp_out = _kp.output['kp']
#
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
# #
# print(kp_out, ki_out/100, kd_out/100)
