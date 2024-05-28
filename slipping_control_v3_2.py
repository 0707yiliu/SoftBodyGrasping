# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Grasping, control force when slipping
import sys
import time

from detection_common.det_common import Det_Common
from detection_common.recording import Camera
# object detection function
from schunk_gripper_common.schunk_gripper_v3 import SchunkGripper
# schunk gripper function
from ur_ikfast import ur_kinematics
import utils
import numpy as np
import rtde_control
import rtde_receive

# !tactile sensor
from sensor_comm_dds.visualisation.visualisers.magtouch_visualiser import MagTouchVisualiser
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from cyclonedds.util import duration

import subprocess
import threading
import pickle
import socket

from scipy.spatial.transform import Rotation as R

from fuzzy_pid import Fuzzy_PID

# schunk BSK tools depends
import os.path

import pyschunk.tools.mylogger
from bkstools.bks_lib.bks_base import keep_communication_alive_input
from bkstools.bks_lib.bks_module import BKSModule, HandleWarningPrintOnly  # @UnusedImport
from bkstools.bks_lib.debug import Print, Var, ApplicationError, g_logmethod  # @UnusedImport


logger = pyschunk.tools.mylogger.getLogger( "BKSTools.demo.demo_grip_workpiece_with_position" )
pyschunk.tools.mylogger.setupLogging()
g_logmethod = logger.info
from pyschunk.generated.generated_enums import eCmdCode
from bkstools.bks_lib import bks_options
# ----------

import yaml

with open('config_bks.yml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f) # import config from yaml



# UR IK solver
config_dir = config['mmdet']['config_dir']
checkpoint_dir = config['mmdet']['checkpoint_dir']
out_dir = config['mmdet']['out_dir']
num_tac_axis = config['magneticSensor']['num_sensor_axis']
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())  # for npy data recording
saved_data = np.zeros((0, num_tac_axis))
offset_sensor_Data = np.zeros(num_tac_axis)
lp_ratio = config['magneticSensor']['lowposs_ratio']
moving_average_window = config['common']['moving_average_window']

# # BSKtools for schunk # ----------------------------------
bks = BKSModule(config['common']['host'],
                sleep_time=None,
                # handle_warning=HandleWarningPrintOnly,
                debug=False,
                repeater_timeout=3.0,
                repeater_nb_tries=5
                )

obj_name = config['common']['obj_name']
obj = str(config['Schunk_UR']['desir_grasp_force']) + obj_name # recorded object
obj = '_' + obj

# # !Camera recording part
record_data = True
record_video = True
stop_record_video = False

# mapping the desired force and step-pos
spmin, spmax = config['Schunk_UR']['pmin'], config['Schunk_UR']['pmax']
Fdmin, Fdmax = config['Schunk_UR']['Fdmin'], config['Schunk_UR']['Fdmax']
mapfunspmin = np.log(1/Fdmax)
mapfunspmax = np.log(1/Fdmin)
# mapping the pos diff and increment force
incre_force_min, incre_force_max = config['Schunk_UR']['increment_force_min'], config['Schunk_UR']['increment_force_max']
pos_diff_min, pos_diff_max = config['Schunk_UR']['pos_diff_min'], config['Schunk_UR']['squeeze_pos_diff']
mapfunincreforcemin = np.log(1/pos_diff_min)
mapfunincreforcemax = np.log(1/pos_diff_max)
def mapping_func(f, xmin, xmax, ymin, ymax):
    # for nonlinear mapping
    sp = np.log(1 / f)
    sp_mapped = (sp - xmin) * (ymax - ymin) / (xmax - xmin) + ymin # linear mapping
    return sp_mapped

# -------------------------

def lowpass_filter(ratio, data, last_data): # online
    data = (1-ratio) * data + ratio * last_data
    return data

def _get_sensor_data():
    sensor_vis = Visualiser(topic_data_type=MagTouch4, description="Visualise data from a MagTouch sensor.")
    # server_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_ip = config['magneticSensor']['local_root_ip']
    # server_port = config['magneticSensor']['server_port']
    # addr = (server_ip, server_port)
    global saved_data, filted_data, offset_sensor_Data
    last_data = np.zeros((2,2,3)).reshape(-1)
    item = 0
    for sample in sensor_vis.reader.take_iter(timeout=duration(seconds=10)):
        start_time = time.time()
        data = np.zeros((2, 2, 3))
        for i, taxel in enumerate(sample.taxels):
            data[i // 2, i % 2] = np.array([taxel.x, taxel.y, taxel.z])
        # print(data, '\n', '---------')
        filted_data = lowpass_filter(lp_ratio, data.reshape(-1), last_data)
        last_data = data.reshape(-1)
        saved_data = np.vstack([saved_data, filted_data - offset_sensor_Data])
        # print(saved_data.shape, time.time() - start_time)
        # if item > 1:
        #     saved_data = np.vstack([saved_data, filted_data - offset_sensor_Data])
        #     item = 0
        #     # print(saved_data.shape)
        # else:
        #     item += 1


def _record_video():
    fps, w, h = (config['common']['record_video_fps'],
                 config['common']['record_video_w'],
                 config['common']['record_video_h'])
    import cv2
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = config['common']['record_video_dir'] + current_time + obj + '.mp4'
    wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)  #
    cam = Camera(w, h, fps)
    global stop_record_video
    while True:
        if record_video is True:
            color_image, depth_image, colorizer_depth = cam.get_frame()
            wr.write(color_image)
            if stop_record_video is True:
                wr.release()
                cam.release()
                break


def recalibrate_tac_sensor(sample_items):
    # zero tactile sensor data
    global filted_data
    offset_sensor_Data = np.zeros(12)
    # colletc data
    for _ in range(sample_items):
        offset_sensor_Data = np.vstack([offset_sensor_Data, filted_data])
        # time.sleep(0.01)
    # mean
    offset_sensor_Data = np.delete(offset_sensor_Data, 0, 0)
    print(offset_sensor_Data.shape)
    offset_sensor_Data.mean(axis=0)
    print('re-calibrate sensor data:', offset_sensor_Data.mean(axis=0))
    return offset_sensor_Data.mean(axis=0)

def _control_loop(
                  filted_data, offset_sensor_Data,
                  _slipping_force, err_z_force_last, err_total,
                  _slipping_force_ratio, _p, _i, _d,):
    tac_z = np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5],
                    filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]])
    tac_big_z = tac_z.max()
    tac_big_z_index = tac_z.argmax()
    err_z_force = _slipping_force - tac_big_z
    d_err = err_z_force - err_z_force_last
    err_total = err_total + err_z_force
    err_z_force_last = err_z_force
    _u = (_p + _slipping_force_ratio) * err_z_force + _i * err_total + _d * d_err

    return err_z_force_last, d_err, err_total, err_z_force, _u


# --------------------------------------------------------
if __name__ == "__main__":
    # # !tactile sensor thread
    sensor_thread = threading.Thread(target=_get_sensor_data)
    sensor_thread.setDaemon(True)
    sensor_thread.start()
    # # !camera recording thread
    cam_thread = threading.Thread(target=_record_video)
    cam_thread.setDaemon(True)
    cam_thread.start()
    # # !RTDE for reading ur, interpreter mode can not use rtde_control
    robot_ip = config['Schunk_UR']['robot_ip']
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    # # !robotic kinamatics model
    ur_arm = ur_kinematics.URKinematics(config['Schunk_UR']['ur_version'])
    # # !create mmdetection model with realsense
    # # # det_comm = Det_Common(config=config_dir, checkpoint=checkpoint_dir, out_pth=out_dir) # TODO:give config file path
    # # # !create schunk gripper
    Print("Make schunk ready...")
    bks.MakeReady()
    move_rel_velocity_ums = int(bks.max_vel * 1000.0)
    force_percent = 100
    # SoftGrip: use the provided velocity:
    # grip_velocity_ums = int(args.grip_velocity * 1000.0)
    grip_direction = BKSModule.grip_from_outside
    move_rel_velocity_ums = int(bks.max_vel * 1000.0)
    # bks.vel = grip_velocity_ums
    print('open finger for initialization.')
    bks.move_to_absolute_position(100, 50000)
    time.sleep(3)
    print('gripper initailization complete')
    # gripper.moveAbsolute(gripper_index, 0.1, init_speed)  # init the position
    _tac_data = np.zeros(num_tac_axis)
    global filted_data
    iter = 0
    gripperDirOut = 'true'
    gripperDirIn = 'false'
    graspforce = config['Schunk_UR']['grip_force_percentage']
    graspspeed = config['Schunk_UR']['grip_speed']

    gripper_pos = np.zeros(1)
    close = True

    grapsing_pos_step = config['Schunk_UR']['grasp_pos_step'] # grasping step (unit: mm)
    # grapsing_pos_step = 2
    _slipping_force = config['Schunk_UR']['desir_grasp_force'] # desired grasping force
    _slipping_force_ratio = 0.5 / _slipping_force * 10
    min_force = config['Schunk_UR']['min_force']
    # force_step = 0.04
    thr = config['Schunk_UR']['grip_thr']
    err_z_force_last = 0.0 # for pid
    err_total = 0.0
    _u = config['fuzzy_pid']['fp_u']
    _p = config['fuzzy_pid']['fp_p']
    _i = config['fuzzy_pid']['fp_i']
    _d = config['fuzzy_pid']['fp_d']


    # if record_video is True:
    #     fps, w, h = (config['common']['record_video_fps'],
    #                  config['common']['record_video_w'],
    #                  config['common']['record_video_h'])
    #     import cv2
    #     mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_path = config['common']['record_video_dir'] + current_time + obj + '.mp4'
    #     wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)  #
    #     cam = Camera(w, h, fps)

    grasp_q = [0.05079088360071182, -1.1178493958762665, 1.5329473654376429, -1.984063287774557, -1.5724676291095179, 0.04206418991088867]
    # grasp_q = [0.041693784296512604, -1.1353824895671387, 1.5678980986224573, -2.0189134083189906, -1.5581014792071741, -0.2504060904132288]
    grasp_q2 = [0.04342854768037796, -1.1026597183993836, 1.6658557097064417, -2.176030775109762, -1.5613611380206507, -0.23878795305360967]
    grasp_q1 = list(np.array([0.04347049072384834, -1.0802181524089356, 1.684894863759176, -2.217555662194723, -1.5615642706500452, -0.23880321184267217]))
    lifting_q1 = list(np.array([0.04331178590655327, -1.1777315002730866, 1.5720866362201136, -2.0071126423277796, -1.5606516043292444, -0.2388375441180628]))
    # lifting_q = [0.04155898839235306, -1.1985772413066407, 1.4263899962054651, -1.814186235467428, -1.557387653981344, -0.25038367906679326]
    lifting_q = [0.050802893936634064, -1.1801475447467347, 1.3791807333575647, -1.7680627308287562, -1.5725005308734339, 0.04205520078539848]
    # control_time = 1.0
    lookahead_time = config['Schunk_UR']['ur_look_ahead_time']
    gain = config['Schunk_UR']['ur_gain']
    desired_slip_force = np.array([_slipping_force])

    rtde_c.servoJ(grasp_q, 0.1, 0.1, 3.0, lookahead_time, gain)
    time.sleep(5)
    print('going to the grasping pos')
    joint_angles_curr = rtde_r.getActualQ()
    target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
    # lifting_step = 0.0008 # 0.8mm
    # -------------------calibrate sensor----------------
    print('going to the zero tac-sensor')
    sample_items = config['magneticSensor']['sample_items']
    offset_sensor_Data = recalibrate_tac_sensor(sample_items)
    print('calibrate sensor complete')
    # ---------------------------------------------------
    grasping = True
    re_grasp = False
    gripin = True
    control = False
    abort = False
    _lifting = False
    reset_lifting_step = False
    simplegrasping = True
    _lift_arm = True
    lifting_force_control = True
    force_control_items = 0
    holdingpos = 0
    _controller_delay = config['fuzzy_pid']['controller_delay']
    # --------------fuzzy pid -----------
    fuzzyPID = Fuzzy_PID()
    pid_items = config['Schunk_UR']['pid_items']
    pid_hoding_times = config['Schunk_UR']['pid_hoding_times']
    current_tac_data = np.zeros(num_tac_axis)
    #--------------lifting config----------
    lift_hz = config['Schunk_UR']['lift_hz']
    lift_wait_times = 0
    lift_history = np.zeros((lift_hz, num_tac_axis))
    d_lift_history = np.zeros((int(lift_hz//2), num_tac_axis))
    o_dy = 0.03
    tac_index = 0 # the index of mainly touching sensor
    control_once = True
    #------- re-grasping params ---------
    det_hz = config['Schunk_UR']['det_hz']
    lift_time = config['Schunk_UR']['lift_time']
    lift_num = 0
    lift_step = config['Schunk_UR']['lift_step'] # unit: meter
    lift_dis = np.copy(target_pos[:3])
    re_items = 0
    _tac_zy = np.zeros(0)
    total_tac_zy = 0
    zy_sum_force_th = config['Schunk_UR']['zy_sum_force_th']
    increment_z_force = config['Schunk_UR']['increment_z_force']
    squeeze_pos_diff = config['Schunk_UR']['squeeze_pos_diff']
    regraspingThr = config['Schunk_UR']['regraspingThr']
    slipdetnum = config['Schunk_UR']['slipdetnum']
    controlspring = config['Schunk_UR']['controlspring']
    first_change = False
    minus_delta_ydz_buffer = 0
    regrasping_times = 0
    minus_delta_ydz_buffer_item = 0
    hard_force = False
    slip_times = 0
    slip_diff_force = 0
    falling_times = 0
    n = 0
    bks.MakeReady()
    while True:
        # det_comm.det_info() # the test mmdetection model
        # !gripper grasping with tactile sensing
        # !setting multiple steps: grasping (close)
        #                          --- control (force control)
        #                          --- lifting (slipping detection)
        #                          --- regrasping from grasping step (inreasing desired force)
        #                          --- done
        try:
            start_time = time.time()
            if re_grasp is True and config['Schunk_UR']['exp_once'] is False:
                # if hard_force is False:
                #     regrasping_times += 1
                #     _slipping_force = config['Schunk_UR']['desir_grasp_force']
                #     # _slipping_force += increment_z_force * regrasping_times
                #     slip_diff_force += ((slip_times + 1) / 1.5) * increment_z_force
                #     _slipping_force += slip_diff_force
                #     print('slipping diff force:', slip_diff_force)
                #     slip_times = 0
                # else:
                #     _slipping_force += increment_z_force
                #     hard_force = False
                _slipping_force += increment_z_force
                rtde_c.servoStop()
                bks.move_to_absolute_position(100, 50000)
                time.sleep(3)
                rtde_c.servoJ(grasp_q, 0.1, 0.1, 3.0, lookahead_time, gain)
                time.sleep(4)
                offset_sensor_Data = recalibrate_tac_sensor(sample_items)
                grasping = True
                re_grasp = False
                control = False
                _lifting = False
                control_once = True
                first_change = False
                simplegrasping = True
                _lift_arm = True
                joint_angles_curr = rtde_r.getActualQ()
                target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                lift_dis = np.copy(target_pos[:3])
                minus_delta_ydz_buffer = 0
                minus_delta_ydz_buffer_item = 0
                bks.MakeReady()
            desired_slip_force = np.append(desired_slip_force, _slipping_force)
            # if simplegrasping is True:
            #     print('grasping step', 'slipping force:', _slipping_force)
            #     bks.set_force = 50  # target force to 50 %
            #     bks.grp_dir = True  # grip from inside
            #     bks.command_code = eCmdCode.MOVE_FORCE
            #     simplegrasping = not simplegrasping

            if grasping is True: # grasping detection
                time.sleep(_controller_delay) # detection hz, fast detection
                # tac_data = np.vstack([tac_data, filted_data - offset_sensor_Data])
                bks.move_to_relative_position(int(0.15*1000.0), move_rel_velocity_ums)
                if _slipping_force > min_force:
                    jug_force = _slipping_force - 0.05
                    # jug_force = min_force
                else:
                    jug_force = _slipping_force
                z_max_force = np.abs([filted_data[2] - offset_sensor_Data[2],
                                      filted_data[5] - offset_sensor_Data[5],
                                      filted_data[8] - offset_sensor_Data[8],
                                      filted_data[11] - offset_sensor_Data[11]])
                if z_max_force.max() > jug_force:
                    z_max_force_index = z_max_force.argmax() # get the index of max z-force
                    # gripper.stop(gripper_index)
                    print('tactile force reached, go to control step')
                    grasping = not grasping
                    control = not control

            if grasping is False and control is True: # z-force holding
                time.sleep(_controller_delay) # controlling frequency
                # !PID controller
                err_z_force_last, d_err, err_total, err_z_force, _u = _control_loop(
                                                                                  filted_data, offset_sensor_Data,
                                                                                  _slipping_force, err_z_force_last, err_total,
                                                                                  _slipping_force_ratio, _p, _i, _d,)
                if _u < 0:
                    _u = controlspring * _u
                _tac_data = np.vstack([_tac_data, filted_data - offset_sensor_Data])
                # !fuzzy control pid parameters
                _p, _i, _d = fuzzyPID.compute(err_z_force, d_err)
                gripper_curr = bks.actual_pos
                # print('current gripper pos:', gripper_curr)
                gripper_pos = np.append(gripper_pos, [gripper_curr])
                # !moving gripper
                # gripper.stop(gripper_index)
                grapsing_pos_step = mapping_func(_slipping_force, xmin=mapfunspmin, xmax=mapfunspmax,
                                                     ymin=spmin, ymax=spmax)
                bks.move_to_relative_position(int(grapsing_pos_step * _u * 1000.0), move_rel_velocity_ums)
                # gripper.moveRelative(gripper_index, grapsing_pos_step * _u, schunk_speed)
                print(err_z_force, d_err, rtde_r.getActualTCPPose()[:3], _slipping_force, _p, _i, _d)

                # !if current force close to desired force calculated by means, then lifting

                if pid_hoding_times > 0:
                    pid_hoding_times -= 1
                else:
                    # !go to lifting part when stabling
                    # if abs(np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5], filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]]).max() - _slipping_force) < thr:
                    if abs(np.array([abs(_tac_data[-pid_items:, 2].mean()),
                                     abs(_tac_data[-pid_items:, 5].mean()),
                                     abs(_tac_data[-pid_items:, 8].mean()),
                                     abs(_tac_data[-pid_items:, 11].mean())]).max() - _slipping_force) < thr:
                        tac_index = np.array([abs(_tac_data[-pid_items:, 2].mean()),
                                              abs(_tac_data[-pid_items:, 5].mean()),
                                              abs(_tac_data[-pid_items:, 8].mean()),
                                              abs(_tac_data[-pid_items:, 11].mean())]).argmax()
                        tac_index = (tac_index + 1) * 3 - 1
                        # pid_items = 10  # for quick check when lifting and slipping
                        # pid_hoding_times = pid_items
                        holdingpos = bks.actual_pos
                        pid_hoding_times = config['Schunk_UR']['pid_hoding_times'] # reset the holding times
                        _lifting = True
                        control = False
                        print('control over, go to lifting step')

            if _lifting is True:
                time.sleep(_controller_delay)
                if control_once is True:
                    control_once = False
                    lifting_force_control = True
                    lift_time_once = config['Schunk_UR']['lift_time_once'] # 50s for once lifting
                    joint_angles_curr = rtde_r.getActualQ()
                    target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                    ori_zpos = np.copy(target_pos[2])
                    target_pos[2] += 0.04  # lift z-axis, lift a minor step once time, hard code
                    target_ideal_rot = [-89.5, 0.1, 0.1]
                    r_target_rot = R.from_euler('xyz', target_ideal_rot, degrees=True)
                    r_target_qua = r_target_rot.as_quat()
                    target_eepos = np.concatenate((target_pos[:3], np.roll(r_target_qua, 1)))
                    target_q = ur_arm.inverse(ee_pose=target_eepos, ee_vec=np.array([0, 0, 0.1507]),
                                              all_solutions=False, q_guess=joint_angles_curr).tolist()
                    if target_q is not None:
                        rtde_c.servoJ(target_q, 0.001, 0.001, lift_time_once, lookahead_time, gain)
                        control_once = False
                    else:
                        print('ik fast inv no response.')
                        control_once = True
                joint_angles_curr = rtde_r.getActualQ()
                curr_zpos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))[2]
                if curr_zpos - ori_zpos > 0.011:
                    force_damp = 0.7
                else:
                    force_damp = 1
                # ------------------------------------
                # if lift_num < (det_hz/2):
                #     lift_num += 1
                # else:
                #     lift_num = 0
                #     joint_angles_curr = rtde_r.getActualQ()
                #     lift_dis[-1] += lift_step
                #     # target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                #     # target_pos[2] = lift_dis # lift z-axis, lift a minor step once time
                #     target_ideal_rot = [-89.5, 0.1, 0.1]
                #     r_target_rot = R.from_euler('xyz', target_ideal_rot, degrees=True)
                #     r_target_qua = r_target_rot.as_quat()
                #     target_eepos = np.concatenate((lift_dis, np.roll(r_target_qua, 1)))
                #     # print(target_eepos)
                #     target_q = ur_arm.inverse(ee_pose=target_eepos, ee_vec=np.array([0, 0, 0.1507]),
                #                               all_solutions=False, q_guess=joint_angles_curr).tolist()
                #     if target_q is not None:
                #         rtde_c.servoJ(target_q, 0.001, 0.001, lift_time, lookahead_time, gain)
                #         control_once = False
                #     else:
                #         print('ik fast inv no response.')
                # ------------------------------------
                re_items += 1
                # the detect hz you need
                _det_hz = int(0.1 * det_hz)
                if re_items > _det_hz: # (_det_hz once time)
                    re_items = 0
                    # data pre-processing
                    #---------------11111111--------------
                    for i in range(num_tac_axis): # smooth each tac point
                        _tac_data[:, i] = utils.moving_average(_tac_data[:, i], moving_average_window)
                        _tac_data[:, i] = utils.FirstOrderLag(_tac_data[:, i], lp_ratio) # for all of the tac data
                    # calculating the related of y/z and delta-y/z in the max z-index
                    regrasping_ydz_related = _tac_data[:, tac_index-1] / _tac_data[:, tac_index]
                    # get real max coupled z-force
                    new_z_force_mean = _tac_data[-_det_hz*3:, tac_index].mean()
                    if _slipping_force < new_z_force_mean:
                        _slipping_force = new_z_force_mean
                    # calculate the derivative of y/z 1second once time
                    # print(tac_index, regrasping_ydz_related[-50:-1], all_tac_data[-50:-1, tac_index], all_tac_data[-50:-1, tac_index-1])
                    regrasping_ydz_related = utils.FirstOrderLag(regrasping_ydz_related, lp_ratio)
                    # one second data for detection
                    delta_ydz = (regrasping_ydz_related[-int(1/_controller_delay/slipdetnum):].mean() -
                                 regrasping_ydz_related[-int(1/_controller_delay/slipdetnum)*2:-int(1/_controller_delay/slipdetnum)].mean())

                    y_mean = _tac_data[-int(1/_controller_delay):, tac_index-1].mean()
                    print('ydz:', delta_ydz, 'y_mean:', y_mean)
                    # joint_angles_curr = rtde_r.getActualQ()
                    # target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                    # print(target_pos[:3])
                    if y_mean <= 0:
                        if delta_ydz < 0: # minus means slipping or falling down
                            minus_delta_ydz_buffer += delta_ydz
                            minus_delta_ydz_buffer_item = 0
                        else:
                            minus_delta_ydz_buffer_item += 1
                            if minus_delta_ydz_buffer_item > 1: # the number of continue holding
                                minus_delta_ydz_buffer_item = 0
                                minus_delta_ydz_buffer = 0
                        if minus_delta_ydz_buffer < -regraspingThr*1.5: # slip detection by derivative of y/z
                            minus_delta_ydz_buffer_item = 0
                            # regrasping or increased force from max z-force
                            print('slipping ---------------------')
                            minus_delta_ydz_buffer = 0
                            _slipping_force += increment_z_force * force_damp
                            slip_times += 1
                    else:
                        if delta_ydz > 0:  # minus means slipping or falling down
                            minus_delta_ydz_buffer += delta_ydz
                            minus_delta_ydz_buffer_item = 0
                        else:
                            minus_delta_ydz_buffer_item += 1
                            if minus_delta_ydz_buffer_item > 1: # the number of continue holding
                                minus_delta_ydz_buffer_item = 0
                                minus_delta_ydz_buffer = 0
                        if minus_delta_ydz_buffer > regraspingThr*1.5:
                            minus_delta_ydz_buffer_item = 0
                            # regrasping or increased force from max z-force
                            print('slipping ---------------------')
                            minus_delta_ydz_buffer = 0
                            _slipping_force += increment_z_force * force_damp
                            slip_times += 1
                    if abs(_tac_data[-int(1/_controller_delay/10):, tac_index].mean()) - (_slipping_force * (1 / 3)) < 0:
                        falling_times += 0.8
                        _slipping_force += (increment_z_force * force_damp / 2) / falling_times

                # control force when slipping -------------------------------
                if lifting_force_control is True:
                    err_z_force_last, d_err, err_total, err_z_force, _u = _control_loop(filted_data,
                                                                                        offset_sensor_Data,
                                                                                        _slipping_force,
                                                                                        err_z_force_last,
                                                                                        err_total,
                                                                                        _slipping_force_ratio,
                                                                                        _p, _i, _d, )
                    _p, _i, _d = fuzzyPID.compute(err_z_force, d_err)
                    gripper_curr = bks.actual_pos
                    grapsing_pos_step = mapping_func(_slipping_force, xmin=mapfunspmin, xmax=mapfunspmax,
                                                     ymin=spmin, ymax=spmax)
                    # create a spring for opening
                    if _u < 0:
                        _u = controlspring * _u * 0.5
                    gripper_des = grapsing_pos_step * _u + gripper_curr
                    print('holding pos:', holdingpos,
                          'gripper des pos:', gripper_des,
                          'des force:', _slipping_force,
                          'curr force err:', err_z_force,
                          'incre force damp:', force_damp)
                    if gripper_des < (holdingpos - 0.5):
                        print('do not open the finger when lifting.')  # do not open the gripper
                    else:
                        # gripper.stop(gripper_index)
                        grapsing_pos_step = mapping_func(_slipping_force, xmin=mapfunspmin, xmax=mapfunspmax,
                                                     ymin=spmin, ymax=spmax)
                        bks.move_to_relative_position(int(grapsing_pos_step * _u * 1000.0), move_rel_velocity_ums)
                        # gripper.moveRelative(gripper_index, grapsing_pos_step * _u / 4, schunk_speed)
                        # print('move???????????')
                    # the gripper position is too hard for all of objects, regrasp
                    if gripper_des - holdingpos > squeeze_pos_diff:  # squeeze 10mm when lifting (too much)
                        re_grasp = True
                        _slipping_force += increment_z_force * force_damp / 3
                        print('regrasping as pos over')
                        hard_force = True
                    else:
                        if 0.001 > gripper_des - holdingpos > -0.001:
                            pdiff = 0.001
                        elif gripper_des - holdingpos < 0:
                            pdiff = 0.005
                        else:
                            pdiff = gripper_des - holdingpos
                        increment_z_force = mapping_func(pdiff,
                                                         xmin=mapfunincreforcemin,
                                                         xmax=mapfunincreforcemax,
                                                         ymin=incre_force_min,
                                                         ymax=incre_force_max)
                    if abs(_tac_data[-int(1/_controller_delay/10):, tac_index].mean()) < (_slipping_force * (1 / 15)):
                        # max z force has going to zero, which means falling down
                        print('falling force:', _tac_data[-int(1/_controller_delay/10):, tac_index].mean(), _slipping_force * (1 / 50), tac_index)
                        print('falling !!!!!!!!!!!!!!!!!!!!!!')
                        _slipping_force += increment_z_force * force_damp / 3
                        re_grasp = True
                        falling_times = 0
# --------------------------Recording Part----------------------------------------------
                _tac_data = np.vstack([_tac_data, filted_data - offset_sensor_Data])

            end_time = time.time()
            # if record_video is True:
            #     color_image, depth_image, colorizer_depth = cam.get_frame()
            #     wr.write(color_image)
        except KeyboardInterrupt:
            stop_record_video = True
            time.sleep(1)
            # if record_video is True:
            #     wr.release()
            #     cam.release()
            if record_data is True:
                _tac_data = np.delete(_tac_data, 0, 0)
                np.savez('./grasp/data/' + current_time + obj + '.npz',
                         loop_tac_data=_tac_data,
                         all_tac_data=saved_data,
                         gripper_pos=gripper_pos,
                         _tac_data=_tac_data,
                         des_slip_force=desired_slip_force,
                         )
            print('keyboard interrupt')
            sys.exit(0)


