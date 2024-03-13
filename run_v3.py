# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping
import sys
import time

from detection_common.det_common import Det_Common
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


def _get_sensor_data():
    sensor_vis = Visualiser(topic_data_type=MagTouch4, description="Visualise data from a MagTouch sensor.")
    server_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_ip = '127.0.0.1'
    server_port = 8110
    addr = (server_ip, server_port)
    global saved_data, filted_data
    for sample in sensor_vis.reader.take_iter(timeout=duration(seconds=10)):
        data = np.zeros((2, 2, 3))
        for i, taxel in enumerate(sample.taxels):
            data[i // 2, i % 2] = np.array([taxel.x, taxel.y, taxel.z])
        # print(data, '\n', '---------')
        filted_data = data.reshape(-1)
        p_data = pickle.dumps(data)
        saved_data = np.vstack([saved_data, filted_data])
        server_udp.sendto(p_data, addr)
        # # mdata = data.copy()
        # time.sleep(0.01)
        # print(data)
        # # print(i)
        # # print(sample.taxels)
        # print('----------')


def _build_sensor_pub():
    # !this function is built for sensor publisher (Useless)
    # !run tactile sensor publisher
    sensor_pub_command = ['python', '-m', 'sensor_comm_dds.communication.readers.magtouch_ble_reader']
    # subprocess.run(sensor_pub_command)
    subprocess.Popen(sensor_pub_command, stdout=subprocess.PIPE)  # TODO:Test it all
    # print('build sensor publisher')


def _get_gripper_pos():
    global gripper_pos
    gripper_pos = np.zeros(1)
    while True:
        gripper_curr = gripper.getPosition()
        gripper_pos = np.append(gripper_pos, [gripper_curr])
        # print(gripper_curr)
        time.sleep(0.01)


# --------------------------------------------------------
if __name__ == "__main__":
    obj = 'banana'  # recorded object
    # # !run tactile sensor publisher
    # sensor_pub = threading.Thread(target=_build_sensor_pub) # useless function
    # sensor_pub.setDaemon(True)
    # sensor_pub.start()
    # !get sensor data
    # sensor_data = np.zeros((2, 2, 3))
    # time.sleep(10)
    sensor_thread = threading.Thread(target=_get_sensor_data)
    sensor_thread.setDaemon(True)
    sensor_thread.start()
    # sensor_data4x3 = MagTouchVisualiser()
    # sensor_data4x3.run()

    # robot_ip = "10.42.0.162"
    # rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    # rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    #
    # # # !create mmdetection model with realsense
    # # det_comm = Det_Common(config=config_dir, checkpoint=checkpoint_dir, out_pth=out_dir) # TODO:give config file path
    # # !create schunk gripper
    local_ip = '10.42.0.111'
    gripper = SchunkGripper(local_ip=local_ip, local_port=44762)
    gripper.connect(remote_function=True)
    gripper_index = 0
    braking = 'true'
    gripper.acknowledge(gripper_index)
    gripper.connect_server_socket()
    gripper_current_pos = gripper.getPosition()
    speed = 100.0
    dir_pos = 0.3
    print('open finger for initialization.')
    gripper.moveAbsolute(gripper_index, 0, speed)  # init the position
    time.sleep(4)

    # # !make subproccess for getting gripper position
    # gripper_pos_thread = threading.Thread(target=_get_gripper_pos)
    # gripper_pos_thread.setDaemon(True)
    # gripper_pos_thread.start()
    # while True:
    #     gripper.moveRelative(gripper_index, dir_pos, speed)

    # joint_space_test()
    # ik_fast_test()

    # !create upd client to receive the tactile sensor date
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local_addr = ("127.0.0.1", 8110)
    udp_socket.bind(local_addr)

    tac_data = np.zeros(12)
    global filted_data
    iter = 0
    gripperDirOut = 'true'
    gripperDirIn = 'false'
    graspforce = 0.6
    graspspeed = 8.7
    gripper.simpleGrip(gripper_index, gripperDirIn, graspforce, graspspeed)
    gripper_pos = np.zeros(0)
    while True:
        # det_comm.det_info() # the test mmdetection model

        # !gripper grasping with tactile sensing
        try:
            iter += 1
            # !control and get gripper's info
            # gripper.moveRelative(gripper_index, dir_pos, speed)  # close gripper
            # # gripper.moveAbsolute(gripper_index, 20, speed)
            # time.sleep(0.1)
            # response = gripper.getPosition()
            # print(response)
            # print('loop iter:', iter)
            # !get tactile sensor's info
            # recv_data = udp_socket.recvfrom(1024)  # get tactile sensor's data
            # info_data = pickle.loads(recv_data[0])
            # print(info_data.reshape(-1))
            tac_data = np.vstack([tac_data, filted_data])
            # gripper_curr = gripper.getPosition()
            gripper_curr = gripper.execute_command(f'EGUEGK_getPosition(1)')
            # print("gripper pos:", gripper_curr)
            time.sleep(0.01)
            # gripper_pos = np.append(gripper_pos, [gripper_curr])
            print('min:', np.min(filted_data))
            if np.min(filted_data) < -1:
                gripper.stop(gripper_index)
                # gripper.simpleGrip(gripper_index, gripperDirOut, graspforce, graspspeed)
                # gripper.stop(gripper_index)
                # tac_data = np.delete(tac_data, 0, 0)
                # saved_data = np.delete(saved_data, 0, 0)
                # np.savez('./grasp/data/' + current_time + '_' + obj + '.npz',
                #          loop_tac_data=tac_data,
                #          all_tac_data=saved_data,
                #          gripper_pos=gripper_pos)
                # # time.sleep(1)
                # # print('go to init pos')
                # # gripper.moveAbsolute(gripper_index, 0.1, speed)
                # # time.sleep(2)
                # # gripper.stop(gripper_index)
                # print('fast stop')
                # gripper.fastStop(gripper_index)
                # udp_socket.close()
                # time.sleep(1)
                # # gripper.stop(gripper_index)
                # # time.sleep(1)
                gripper.disconnect()
                print('force stop exit')
                sys.exit(0)
            # time.sleep(0.1)
            # print(tac_data.shape)
            # print('----')
        except KeyboardInterrupt:
            gripper.stop(gripper_index)
            gripper.fastStop(gripper_index)
            tac_data = np.delete(tac_data, 0, 0)
            saved_data = np.delete(saved_data, 0, 0)
            np.savez('./grasp/data/' + current_time + '_' + obj + '.npz',
                     loop_tac_data=tac_data,
                     all_tac_data=saved_data,
                     gripper_pos=gripper_pos)
            udp_socket.close()
            time.sleep(1)
            # gripper.stop(gripper_index)
            # time.sleep(1)
            gripper.disconnect()
            print('keyboard interrupt')