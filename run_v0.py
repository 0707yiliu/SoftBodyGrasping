# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping

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

import subprocess
import threading

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
    sensor_data4x3 = MagTouchVisualiser()
    sensor_data4x3.run()
    sensor_data = sensor_data4x3.data # TODO:get the data in this loop

# --------------------------------------------------------
if __name__ == "__main__":
    # !run tactile sensor publisher
    sensor_pub_command = ['python', 'sensor_comm_dds/communication/readers/magtouch_ble_reader.py']
    # subprocess.run(sensor_pub_command)
    sensor_process = subprocess.Popen(sensor_pub_command, stdout=subprocess.PIPE)  # TODO:Test it all
    # !get sensor data
    sensor_data = np.zeros((2, 2, 3))
    sensor_thread = threading.Thread(target=_get_sensor_data)
    sensor_thread.setDaemon(True)
    sensor_thread.start()
    # sensor_data4x3 = MagTouchVisualiser()
    # sensor_data4x3.run()

    robot_ip = "10.42.0.162"
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)

    # # !create mmdetection model with realsense
    # det_comm = Det_Common(config=config_dir, checkpoint=checkpoint_dir, out_pth=out_dir) # TODO:give config file path
    # # !create schunk gripper
    gripper = SchunkGripper(local_port=44875)
    gripper.connect()
    gripper_index = 0
    gripper.acknowledge(gripper_index)
    gripper.connect_server_socket()
    gripper_current_pos = gripper.getPosition()
    speed = 30.0
    dir_pos = 0.1
    while True:
        gripper.moveRelative(gripper_index, dir_pos, speed)

    joint_space_test()
    ik_fast_test()
    # while True:
    #     det_comm.det_info()
