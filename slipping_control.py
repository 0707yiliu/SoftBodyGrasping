# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping
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

# UR IK solver
config_dir = '/home/yi/mmdet_models/configs/defobjs/mask-rcnn_r101_fpn_ms-poly-3x_defobjs_20.py'
checkpoint_dir = '/home/yi/mmdet_models/checkpoints/mask-rcnn_r101_fpn_ms-poly-3x_defobjs_20/epoch_10.pth'
out_dir = '/home/yi/mmdet_models/out.video'
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())  # for npy data recording
saved_data = np.zeros(12)


#
# # example of ur ikfast -------------
def ik_fast_test():
    ur_arm = ur_kinematics.URKinematics('ur3e')
    joint_angles = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.]  # in radians
    print("joint angles", joint_angles)
    pose_quat = ur_arm.forward(joint_angles)
    pose_matrix = ur_arm.forward(joint_angles, 'matrix')
    print("forward() quaternion \n", pose_quat)
    print("forward() matrix \n", pose_matrix)
    # print("inverse() all", ur3e_arm.inverse(pose_quat, True))
    print("inverse() one from quat", ur_arm.inverse(pose_quat, False, q_guess=joint_angles))
    print("inverse() one from matrix", ur_arm.inverse(pose_matrix, False, q_guess=joint_angles))


# # -----------------------------------
# ---------- Joint-Space interpolate TEST ------------
def joint_space_test():
    # define the velocity you need in each point, and you can calculate the time you need (default time = 1)
    start_angle = np.array([1, 2, 3, 4, 5, 6])
    end_angle = start_angle * 5
    time = 1
    num_points = 10
    start_vel = np.array([0, 0, 0, 0, 0, 0])
    end_vel = np.array([1, 1, 1, 1, 1, 1])
    start_acc = start_vel
    end_acc = start_vel
    q, v, a = utils.joint_space_interpolate_quintic(
        start_angle, end_angle,
        time,
        start_vel, end_vel,
        start_acc, end_acc,
        num_points
    )
    print(q, '\n', v, '\n', a)

def lowpass_filter(ratio, data, last_data): # online
    data = ratio * data + (1-ratio) * last_data
    return data

def FirstOrderLag(inputs, a): # offline
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs

def _get_sensor_data():
    sensor_vis = Visualiser(topic_data_type=MagTouch4, description="Visualise data from a MagTouch sensor.")
    server_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_ip = '127.0.0.1'
    server_port = 8110
    addr = (server_ip, server_port)
    lp_ratio = 0.3
    global saved_data, filted_data
    last_data = np.zeros((2,2,3)).reshape(-1)
    for sample in sensor_vis.reader.take_iter(timeout=duration(seconds=10)):
        data = np.zeros((2, 2, 3))
        for i, taxel in enumerate(sample.taxels):
            data[i // 2, i % 2] = np.array([taxel.x, taxel.y, taxel.z])
        # print(data, '\n', '---------')
        last_data = data.reshape(-1)
        filted_data = lowpass_filter(lp_ratio, data.reshape(-1), last_data)



def _build_sensor_pub():
    # !this function is built for sensor publisher (Useless)
    # !run tactile sensor publisher
    sensor_pub_command = ['python', '-m', 'sensor_comm_dds.communication.readers.magtouch_ble_reader']

    subprocess.Popen(sensor_pub_command, stdout=subprocess.PIPE)  # TODO:Test it all



def _get_gripper_pos():
    global gripper_pos
    gripper_pos = np.zeros(1)
    while True:
        gripper_curr = gripper.getPosition()
        gripper_pos = np.append(gripper_pos, [gripper_curr])
        # print(gripper_curr)
        time.sleep(0.01)

def recalibrate_tac_sensor(sample_items):
    global filted_data
    offset_sensor_Data = np.zeros(12)
    loop_items = 1000
    # colletc data
    for _ in range(sample_items):
        offset_sensor_Data = np.vstack([offset_sensor_Data, filted_data])
        # time.sleep(0.01)
    # mean
    offset_sensor_Data = np.delete(offset_sensor_Data, 0, 0)
    print(offset_sensor_Data.shape)
    offset_sensor_Data.mean(axis=0)
    print('re-calibrate sensor data:', offset_sensor_Data.mean(axis=0))
    return  offset_sensor_Data.mean(axis=0)

# --------------------------------------------------------
if __name__ == "__main__":

    # !Camera recording part
    record_data = True
    record_video = True

    sensor_thread = threading.Thread(target=_get_sensor_data)
    sensor_thread.setDaemon(True)
    sensor_thread.start()
    tac_th_z = -5
    max_pos = 63

    RTDE = True
    if RTDE is True:
        robot_ip = "10.42.0.162"
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    ur_arm = ur_kinematics.URKinematics('ur3e')
    #
    # # # !create mmdetection model with realsense
    # # det_comm = Det_Common(config=config_dir, checkpoint=checkpoint_dir, out_pth=out_dir) # TODO:give config file path
    # # !create schunk gripper
    local_ip = '10.42.0.111'
    gripper = SchunkGripper(local_ip=local_ip, local_port=44607)
    gripper.connect(remote_function=True)
    gripper_index = 0
    braking = 'true'
    gripper.acknowledge(gripper_index)
    gripper.connect_server_socket()
    gripper_current_pos = gripper.getPosition()
    init_speed = 100
    schunk_speed = 10
    dir_pos = 0.3
    print('open finger for initialization.')
    gripper.moveAbsolute(gripper_index, 0.1, init_speed)  # init the position
    time.sleep(4)
    print("gripper initailization complete")

    tac_data = np.zeros(12)
    _tac_data = np.zeros(12)
    global filted_data
    iter = 0
    gripperDirOut = 'true'
    gripperDirIn = 'false'
    graspforce = 0.51
    graspspeed = 6

    gripper_pos = np.zeros(1)
    close = True

    grapsing_pos_step = 0.1
    # grapsing_pos_step = 2
    _slipping_force = 0.05
    _slipping_force_ratio = 0.5 / _slipping_force * 10
    force_step = 0.04
    thr = 0.02
    err_z_force_last = 0.0 # for pid
    err_total = 0.0
    _u = 0 # pid
    _p = 1
    _i = 0.005
    _d = 0.005
    obj = str(_slipping_force) + 'force_lego_lift' + str(_p) + '-' + str(_i) + '-' + str(_d)  # recorded object
    obj = '_' + obj
    if record_video is True:
        fps, w, h = 30, 1280, 720
        import cv2
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = '/home/yi/robotic_manipulation/_graspdata/sliding/' + current_time + obj + '.mp4'
        wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)  #
        cam = Camera(w, h, fps)
    grasping = True
    grasp_q = [0.05079088360071182, -1.1178493958762665, 1.5329473654376429, -1.984063287774557, -1.5724676291095179, 0.04206418991088867]
    # grasp_q = [0.041693784296512604, -1.1353824895671387, 1.5678980986224573, -2.0189134083189906, -1.5581014792071741, -0.2504060904132288]
    grasp_q2 = [0.04342854768037796, -1.1026597183993836, 1.6658557097064417, -2.176030775109762, -1.5613611380206507, -0.23878795305360967]
    grasp_q1 = list(np.array([0.04347049072384834, -1.0802181524089356, 1.684894863759176, -2.217555662194723, -1.5615642706500452, -0.23880321184267217]))
    lifting_q1 = list(np.array([0.04331178590655327, -1.1777315002730866, 1.5720866362201136, -2.0071126423277796, -1.5606516043292444, -0.2388375441180628]))
    # lifting_q = [0.04155898839235306, -1.1985772413066407, 1.4263899962054651, -1.814186235467428, -1.557387653981344, -0.25038367906679326]
    lifting_q = [0.050802893936634064, -1.1801475447467347, 1.3791807333575647, -1.7680627308287562, -1.5725005308734339, 0.04205520078539848]
    control_time = 1.0
    lookahead_time = 0.03
    gain = 800.0

    gripper.servoJ(grasp_q, 0.1, 0.1, 3.0, lookahead_time, gain)
    time.sleep(4)
    print('going to the grasping pos')
    joint_angles_curr = rtde_r.getActualQ()
    target_pos = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
    lifting_step = 0.0008 # 0.8mm
    # -------------------calibrate sensor----------------
    print('going to the zero tac-sensor')
    sample_items = 30000
    offset_sensor_Data = recalibrate_tac_sensor(sample_items)
    print('calibrate sensor complete')
    # ---------------------------------------------------
    gripin = True
    control = False
    stay_item = 300
    abort = False
    _lifting = False
    reset_lifting_step = False
    _controller_delay = 0.5
    # --------------fuzzy pid -----------
    fuzzyPID = Fuzzy_PID()
    pid_items = 10
    pid_hoding_times = pid_items
    current_tac_data = np.zeros(12)
    #--------------lifting config----------
    lift_hz = 50
    lift_wait_times = 0
    lift_history = np.zeros((lift_hz, 12))
    d_lift_history = np.zeros((int(lift_hz//2), 12))
    o_dy = 0.03
    tac_index = 0 # the index of mainly touching sensor
    control_once = True
    while True:
        # det_comm.det_info() # the test mmdetection model
        # !gripper grasping with tactile sensing
        try:
            start_time = time.time()

            if grasping is True:
                # !move robot to the grasping pos
                time.sleep(_controller_delay)
                tac_data = np.vstack([tac_data, filted_data-offset_sensor_Data])
                gripper.moveRelative(gripper_index, grapsing_pos_step, schunk_speed)
                gripper_curr = gripper.getPosition()
                print('current gripper pos:', gripper_curr)
                gripper_pos = np.append(gripper_pos, [gripper_curr])
                if _slipping_force > 0.13:
                    jug_force = _slipping_force - 0.08
                else:
                    jug_force = _slipping_force
                print(filted_data-offset_sensor_Data)
                if np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5], filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]]).max() > jug_force:
                    print('tactile force reached')
                    grasping = False
                    control = True
            if grasping is False and control is True:
                time.sleep(_controller_delay)
                tac_z = np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5],
                                filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]])
                tac_big_z = tac_z.max()
                tac_big_z_index = tac_z.argmax()
                err_z_force = _slipping_force - tac_big_z
                d_err = err_z_force - err_z_force_last
                # p_err = _slipping_force - tac_big_z
                err_total = err_total + err_z_force
                err_z_force_last = err_z_force
                _u = (_p + _slipping_force_ratio) * err_z_force + _i * err_total + _d * d_err
                _tac_data = np.vstack([_tac_data, filted_data - offset_sensor_Data])
                pid_means_items = pid_items + 10
                fuzzy_kp, fuzzy_ki, fuzzy_kd = fuzzyPID.compute(err_z_force, d_err)
                # print(fuzzy_kp, fuzzy_ki, fuzzy_kd)
                _p, _i, _d = fuzzyPID.compute(err_z_force, d_err)
                gripper.stop(gripper_index)
                gripper.moveRelative(gripper_index, grapsing_pos_step * _u, schunk_speed)
                print(err_z_force, d_err, rtde_r.getActualTCPPose()[:3], _slipping_force, _p, _i, _d)
                gripper_curr = gripper.getPosition()
                print('current gripper pos:', gripper_curr)
                gripper_pos = np.append(gripper_pos, [gripper_curr])
                if _tac_data.shape[0] > pid_means_items:

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
                            pid_items = 3  # for quick check when lifting and slipping
                            pid_hoding_times = pid_items
                            _lifting = True
                            control = False
                        else:
                            # !get the new number of pid_items for entering lifting part
                            if reset_lifting_step is True:
                                _slipping_force += force_step
                                reset_lifting_step = False
            if _lifting is True:
                # # print('lifting')
                # ------------------------check and goback to fuzzy control loop---------------------------
                # # joint_angles_curr = rtde_r.getActualQ()
                # # pose_quat = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                # # target_pos = pose_quat
                # target_pos[2] += lifting_step
                # print(target_pos[:3])
                # target_ideal_rot = [-89.5, 0.1, 0.1]
                # r_target_rot = R.from_euler('xyz', target_ideal_rot, degrees=True)
                # r_target_qua = r_target_rot.as_quat()
                # target_eepos = np.concatenate((target_pos[:3], np.roll(r_target_qua, 1)))
                # target_q = ur_arm.inverse(ee_pose=target_eepos, ee_vec=np.array([0, 0, 0.1507]),
                #                           all_solutions=False, q_guess=joint_angles_curr)
                # print('joint init q:', joint_angles_curr)
                # print('joint target q:', target_q)
                # if target_q is not None:
                #     gripper.servoJ(target_q.tolist(), 0.1, 0.1, control_time, lookahead_time, gain)
                #     _lifting = False
                #     control = True
                #     reset_lifting_step = True
                #     # time.sleep(control_time)
                #     # _slipping_force += force_step
                # else:
                #     print('ik fast inv no response.')
                # ----------------------------------------------------------------------------------------
                # -----------------------------------simple lifting----------------------------------
                # !move robot to the lifting end pos
                time.sleep(0.02)
                lower_move_time = 50
                if control_once is True:
                    gripper.servoJ(lifting_q, 0.1, 0.1, lower_move_time, lookahead_time, gain)
                    control_once = False
                # -------------------------------------------------------------------------------------
                # ------------------lifting holding x,y,z without PID control--------------------
                # !this part of time sleep is 50hz, we open the tactile sensor's sampling frequency for lifting
                # time.sleep(1/lift_hz) # sampling number (lift_hz)
                # if lift_wait_times < lift_hz:
                #     lift_history[lift_wait_times, :] = filted_data - offset_sensor_Data
                #     lift_wait_times += 1
                # else:
                #     lift_wait_times = 0
                #     # calculate derivative
                #     for i in range(d_lift_history.shape[0]):
                #         d_lift_history[i, :] = (lift_history[-i+1, :] - lift_history[-i+2, :]) / (1/lift_hz)
                #     d_lift_history_means = d_lift_history.mean(0)
                #     print('-------------')
                #     print(tac_index)
                #     print(d_lift_history_means)
                #     # !check dy is stable around 0
                #     # !if yes, lift a minor step
                #     # !if no, squeeze a minor step
                #     if abs(d_lift_history_means[tac_index * 3 + 1]) < o_dy:
                #     # if np.all(np.abs([d_lift_history_means[1], d_lift_history_means[4],
                #     #                   d_lift_history_means[7], d_lift_history_means[10]]) < o_dy):
                #         # stable and set lifting
                #         joint_angles_curr = rtde_r.getActualQ()
                #         pose_quat = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
                #         target_pos = pose_quat
                #         target_pos[2] += lifting_step  # lift a minor step once time
                #         target_ideal_rot = [-89.5, 0.1, 0.1]
                #         r_target_rot = R.from_euler('xyz', target_ideal_rot, degrees=True)
                #         r_target_qua = r_target_rot.as_quat()
                #         target_eepos = np.concatenate((target_pos[:3], np.roll(r_target_qua, 1)))
                #         _tac_data = np.vstack([_tac_data, filted_data - offset_sensor_Data])
                #         target_q = ur_arm.inverse(ee_pose=target_eepos, ee_vec=np.array([0, 0, 0.1507]),
                #                                   all_solutions=False, q_guess=joint_angles_curr)
                #         if target_q is not None:
                #             print('lift')
                #             gripper.servoJ(target_q.tolist(), 0.1, 0.1, control_time, lookahead_time, gain)
                #         else:
                #             print('ik fast inv no response.')
                #     else:
                #         print('squeeze')
                #         gripper.moveRelative(gripper_index, grapsing_pos_step, schunk_speed)
                # -------------------------------------------------------------------------------------
                _tac_data = np.vstack([_tac_data, filted_data - offset_sensor_Data])

            end_time = time.time()
            if record_video is True:
                color_image, depth_image, colorizer_depth = cam.get_frame()
                wr.write(color_image)
        except KeyboardInterrupt:
            # time.sleep(1)
            if record_video is True:
                wr.release()
                cam.release()
            if record_data is True:
                tac_data = np.delete(tac_data, 0, 0)
                _tac_data = np.delete(_tac_data, 0, 0)
                saved_data = np.delete(saved_data, 0, 0)
                np.savez('./grasp/data/' + current_time + obj + '.npz',
                         loop_tac_data=tac_data,
                         all_tac_data=saved_data,
                         gripper_pos=gripper_pos,
                         _tac_data=_tac_data,
                         )
            gripper.stop(gripper_index)
            gripper.fastStop(gripper_index)

            gripper.execute_command("skipbuffer")
            gripper.execute_command("abort")
            gripper.disconnect()
            gripper.close_socket()

            # udp_socket.close()
            time.sleep(1)
            # gripper.stop(gripper_index)
            # time.sleep(1)

            print('keyboard interrupt')
            sys.exit(0)


