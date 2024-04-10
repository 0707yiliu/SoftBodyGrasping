# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping
import time

from detection_common.det_common import Det_Common
# object detection function
from schunk_gripper_common.schunk_gripper_v2 import SchunkGripper
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

# UR IK solver
config_dir = '/home/yi/mmdet_models/configs/defobjs/mask-rcnn_r101_fpn_ms-poly-3x_defobjs_20.py'
checkpoint_dir = '/home/yi/mmdet_models/checkpoints/mask-rcnn_r101_fpn_ms-poly-3x_defobjs_20/epoch_10.pth'
out_dir = '/home/yi/mmdet_models/out.video'
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
    for sample in sensor_vis.reader.take_iter(timeout=duration(seconds=10)):
        data = np.zeros((2, 2, 3))
        for i, taxel in enumerate(sample.taxels):
            data[i // 2, i % 2] = np.array([taxel.x, taxel.y, taxel.z])
        print(data, '\n', '---------')
        p_data = pickle.dumps(data)
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

# --------------------------------------------------------
if __name__ == "__main__":
    robot_ip = "10.42.0.162"
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    ur_arm = ur_kinematics.URKinematics('ur3e')
    joint_angles_curr = rtde_r.getActualQ()
    print("joint angles:", joint_angles_curr)
    print('real quaternion:', rtde_r.getActualTCPPose())
    pose_quat = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]))
    pose_matrix = ur_arm.forward(joint_angles_curr, ee_vec=np.array([0, 0, 0.1507]), rotation_type='matrix')
    print("forward() quaternion \n", pose_quat, type(pose_quat))
    print("forward() matrix \n", pose_matrix, type(pose_matrix))

    target_pos = pose_quat
    target_pos[2] -= 0

    ideal_quat = np.roll(pose_quat[3:], -1) # to xyzw
    ideal_quatR = R.from_quat(ideal_quat)
    print(ideal_quatR.as_euler('xyz', degrees=True))
    # current_ideal_rot = ideal_quatR.as_euler('xyz', degrees=True)
    # [-73.01539164 - 0.89202353 - 0.74797427]
    target_ideal_rot = [-89.5, 0.1, 0.1]
    r_target_rot = R.from_euler('xyz', target_ideal_rot, degrees=True)
    r_target_qua = r_target_rot.as_quat()
    print('---------------------', r_target_qua)
    target_pos1 = np.concatenate((target_pos[:3], np.roll(r_target_qua, 1)))
    print(target_pos1, "\n", target_pos)


    # print("inverse() all", ur3e_arm.inverse(pose_quat, True))
    print("inverse() one from quat", ur_arm.inverse(ee_pose=target_pos, ee_vec=np.array([0, 0, 0.1507]),
                                                    all_solutions=False, q_guess=joint_angles_curr))
    print("inverse() one from matrix", ur_arm.inverse(ee_pose=pose_matrix, ee_vec=np.array([0, 0, 0.1507]),
                                                      all_solutions=False, q_guess=joint_angles_curr))
    print("inverse() one from quat", ur_arm.inverse(ee_pose=target_pos1, ee_vec=np.array([0, 0, 0.1507]),
                                                    all_solutions=False, q_guess=joint_angles_curr))

    target_q = ur_arm.inverse(ee_pose=target_pos1, ee_vec=np.array([0, 0, 0.1507]),
                              all_solutions=False, q_guess=joint_angles_curr)
    if target_q is not None:
        rtde_c.servoJ(target_q, 0.1, 0.1, 5, 0.03, 800)


