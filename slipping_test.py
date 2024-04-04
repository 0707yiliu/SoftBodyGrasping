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
    # obj = '0_1force_silicone'  # recorded object
    # obj = '_' + obj
    # !Camera recording part
    record_data = True
    record_video = True


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
    tac_th_z = -5
    max_pos = 63
    # sensor_data4x3 = MagTouchVisualiser()
    # sensor_data4x3.run()
    RTDE = True
    if RTDE is True:
        robot_ip = "10.42.0.162"
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
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
    speed = 100.0
    dir_pos = 0.3
    print('open finger for initialization.')
    gripper.moveAbsolute(gripper_index, 0.1, speed)  # init the position
    time.sleep(4)
    print("gripper initailization complete")

    # # !make subproccess for getting gripper position
    # gripper_pos_thread = threading.Thread(target=_get_gripper_pos)
    # gripper_pos_thread.setDaemon(True)
    # gripper_pos_thread.start()
    # while True:
    #     gripper.moveRelative(gripper_index, dir_pos, speed)

    # joint_space_test()
    # ik_fast_test()

    # !create upd client to receive the tactile sensor date
    # udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # local_addr = ("127.0.0.1", 8110)
    # udp_socket.bind(local_addr)

    tac_data = np.zeros(12)
    _tac_data = np.zeros(12)
    global filted_data
    iter = 0
    gripperDirOut = 'true'
    gripperDirIn = 'false'
    graspforce = 0.51
    graspspeed = 8.1
    # gripper.simpleGrip(gripper_index, gripperDirIn, graspforce, graspspeed)
    # gripper.moveAbsolute(gripper_index, 50, graspspeed)
    gripper_pos = np.zeros(1)
    close = True
    # gripper_curr = gripper.getPosition()
    # # gripper_curr = gripper.execute_command(f'EGUEGK_getPosition(1)')
    # time.sleep(0.01)
    # gripper.waitForComplete(gripper_index, timeout=100)
    # time.sleep(0.01)
    grapsing_pos_step = 0.3
    _slipping_force = 0.1
    obj = str(_slipping_force) + 'force_cup'  # recorded object
    obj = '_' + obj
    if record_video is True:
        fps, w, h = 30, 1280, 720
        import cv2
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = './grasp/data/' + current_time + obj + '.mp4'
        wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)  #
        cam = Camera(w, h, fps)
    grasping = True
    grasp_q = [0.041693784296512604, -1.1353824895671387, 1.5678980986224573, -2.0189134083189906, -1.5581014792071741, -0.2504060904132288]
    grasp_q2 = [0.04342854768037796, -1.1026597183993836, 1.6658557097064417, -2.176030775109762, -1.5613611380206507, -0.23878795305360967]
    grasp_q1 = list(np.array([0.04347049072384834, -1.0802181524089356, 1.684894863759176, -2.217555662194723, -1.5615642706500452, -0.23880321184267217]))
    lifting_q1 = list(np.array([0.04331178590655327, -1.1777315002730866, 1.5720866362201136, -2.0071126423277796, -1.5606516043292444, -0.2388375441180628]))
    lifting_q = [0.04155898839235306, -1.1985772413066407, 1.4263899962054651, -1.814186235467428, -1.557387653981344, -0.25038367906679326]
    control_time = 5.0
    lookahead_time = 0.03
    gain = 500.0

    gripper.servoJ(grasp_q, 0.1, 0.1, control_time, lookahead_time, gain)
    time.sleep(5)
    print('going to the grasping pos')
    # -------------------calibrate sensor----------------
    sample_items = 1000
    offset_sensor_Data = recalibrate_tac_sensor(sample_items)
    # print('calibrate sensor complete')
    # ---------------------------------------------------
    gripin = True
    stay_item = 500
    while True:
        # det_comm.det_info() # the test mmdetection model

        # !gripper grasping with tactile sensing
        try:
            # gripper_curr = gripper.getPosition()
            # if RTDE is True:
            #     # pass
            #     print(rtde_r.getActualQ())
                # !------------------ur script sent by interpreter mode-----------
                # gripper.servoJ_inter(grasp_q, 0.1, 0.1, control_time, lookahead_time, gain)
            # !---------------ur script mode------------------
            # gripper.servoJ(grasp_q, 0.1, 0.1, control_time, lookahead_time, gain)
            # print('moving to grasping pos')
            # time.sleep(5)
            # gripper.simpleGrip(gripper_index, gripperDirIn, graspforce, graspspeed)
            # print('gripper in')
            # time.sleep(5)
            # gripper.servoJ(lifting_q, 0.1, 0.1, control_time, lookahead_time, gain)
            # print('moving to lifting pos')
            # time.sleep(5)
            # gripper.simpleGrip(gripper_index, gripperDirOut, graspforce, graspspeed)
            # print('gripper out')
            # time.sleep(5)
            # it works when sending the command to the interpreter mode port
                # -------------------------------- !!!!!!!!!!!
                # rtde_c.servoJ(grasp_q, 0.1, 0.1, control_time, lookahead_time, gain)
                # print('moving to grasping pos')
                # time.sleep(10)
                # rtde_c.servoJ(lifting_q, 0.1, 0.1, control_time, lookahead_time, gain)
                # print('moving to lifting pos')
                # time.sleep(10)
                # RTDE controlling functions are not work with interpreter mode for schunk gripper

            time.sleep(0.01)
            if grasping is True:
                # !move robot to the grasping pos
                # time.sleep(5)
                # rtde_c.servoJ(q=grasp_q, time=control_time, lookahead_time=lookahead_time, gain=gain) # TODO: control the manually joint setting
                # !grasping with step force
                # time.sleep(1)
                # time.sleep(0.01)
                # gripper.moveRelative(gripper_index, grapsing_pos_step, 10)
                if gripin is True:
                    gripper.simpleGrip(gripper_index, gripperDirIn, graspforce, graspspeed)
                    gripin = False
                tac_data = np.vstack([tac_data, filted_data-offset_sensor_Data])

                print(np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5], filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]]).max())
                if np.abs([filted_data[2] - offset_sensor_Data[2], filted_data[5] - offset_sensor_Data[5], filted_data[8] - offset_sensor_Data[8], filted_data[11] - offset_sensor_Data[11]]).max() > _slipping_force:
                # print(np.abs(filted_data-offset_sensor_Data).max())
                # if np.abs(filted_data-offset_sensor_Data).max() > _slipping_force:
                    gripper.stop(gripper_index)
                    time.sleep(0.01)
                    print('tactile force reached')
                    grasping = False
            else:
                # ntrp.execute_command("skipbuffer")
                # logging.info(f"Last command executing before skipbuffer: {intrp.get_last_executed_id()}")
                #
                # logging.info("Aborting running move command")
                # intrp.execute_command("abort")
                gripper.execute_command("skipbuffer")
                gripper.execute_command("abort")
                if stay_item > 0:
                    stay_item -= 1
                else:
                    # !move robot to the lifting end pos
                    gripper.servoJ(lifting_q, 0.1, 0.1, control_time, lookahead_time, gain)
                # rtde_c.servoJ(q=lifting_q, time=control_time, lookahead_time=lookahead_time, gain=gain) # TODO: control the manually joint setting
                _tac_data = np.vstack([_tac_data, filted_data-offset_sensor_Data])

            # # print(gripper_curr)
            # # time.sleep(0.1)
            # # iter += 1
            # gripper_curr = gripper.getPosition()
            # # gripper.execute_command(f'EGUEGK_getPosition(0)')
            # # print('gripper pos:', gripper_curr)
            # time.sleep(0.01) # 100hz is ok, 200hz a little bit fast
            # # !for camera recording
            if record_video is True:
                color_image, depth_image, colorizer_depth = cam.get_frame()
                wr.write(color_image)
            # ------------------------------------------------------------------

            # # print('iter:', iter)
            # # end = False
            # # th = 50
            # # if iter < th:
            # #     yes = False
            # #     gripper.moveRelative(gripper_index, 1, 100)
            # #     # gripper.moveAbsolute(gripper_index, 50, graspspeed)
            # #     # gripper.waitForComplete(gripper_index, timeout=1)
            # # if iter > th and end is False:
            # #     gripper.moveRelative(gripper_index, -1, 100)
            # #     # gripper.moveAbsolute(gripper_index, 0, graspspeed)
            # #     # gripper.waitForComplete(gripper_index, timeout=1)
            # #     yes = True
            # # if yes is True and gripper_curr < 10:
            # #     end = True
            # #     gripper.stop(gripper_index)
            # # gripper.moveRelative(gripper_index, -2, graspspeed)
            # # # time.sleep(0.01)
            # # gripper.moveAbsolute(gripper_index, -1, graspspeed)
            # # time.sleep(0.01)
            # # gripper.moveAbsolute(gripper_index, 50, graspspeed)
            # # time.sleep(0.01)
            # # gripper.moveAbsolute(gripper_index, 0, graspspeed)
            # # time.sleep(0.01)
            # # gripper_curr = gripper.getPosition()
            # # print('global gripper pos:', gripper_curr)
            # # time.sleep(0.01)
            #
            # gripper_pos = np.append(gripper_pos, [gripper_curr])
            #
            # if type(gripper_curr) == type(0.1) and close is True:
            #     # print('close')
            #     # gripper.stop(gripper_index)
            #     # gripper.moveAbsolute(gripper_index, 50, graspspeed)
            #     # gripper.moveRelative(gripper_index, 1, graspspeed)
            #     # # time.sleep(0.01)
            #     # iter += 1
            #     # tac_data = np.vstack([tac_data, filted_data])
            #     # gripper_curr = gripper.getPosition()
            #     # gripper_pos = np.append(gripper_pos, [gripper_curr])
            #     # gripper_curr = gripper.execute_command(f'EGUEGK_getPosition(1)')
            #     # print("gripper pos:", gripper_curr, type(gripper_curr))
            #     # time.sleep(0.01)
            #     # print('min:', np.min(filted_data))
            #     tmp_tac_data = np.array([filted_data[2], filted_data[5], filted_data[8], filted_data[11]])
            #     # print(tmp_tac_data)
            #     # print('filted data:', filted_data)
            #     if gripper_curr > max_pos or tmp_tac_data.min() < tac_th_z:
            #         print('stop')
            #         close = False
            #         print('close:', gripper_curr, close is False, type(gripper_curr) == type(0.1) and close is True)
            #         gripper.stop(gripper_index)
            #         time.sleep(0.01)
            #         gripper.simpleGrip(gripper_index, gripperDirOut, graspforce, graspspeed)
            # if close is False:
            #     # print("here!!!!!!!!!!!!!")
            #     # gripper.stop(gripper_index)
            #     # gripper.moveAbsolute(gripper_index, 10, graspspeed)
            #     # gripper.moveRelative(gripper_index, -1, graspspeed)
            #     # time.sleep(1)
            #     # iter += 1
            #     # # tac_data = np.vstack([tac_data, filted_data])
            #     # gripper_curr = gripper.getPosition()
            #     # gripper_pos = np.append(gripper_pos, [gripper_curr])
            #     # # gripper_curr = gripper.execute_command(f'EGUEGK_getPosition(1)')
            #     # print("gripper pos:", gripper_curr)
            #     # time.sleep(0.01)
            #     # # print('min:', np.min(filted_data)
            #
            #     # !return part
            #     # if gripper_curr < 20:
            #     #     close = True
            #     #     gripper.stop(gripper_index)
            #     #     time.sleep(0.01)
            #     #     gripper.simpleGrip(gripper_index, gripperDirIn, graspforce, graspspeed)
            #
            #     # !stop part
            #     if gripper_curr < 2:
            #         if record_video is True:
            #             wr.release()
            #             cam.release()
            #         if record_data is True:
            #             tac_data = np.delete(tac_data, 0, 0)
            #             saved_data = np.delete(saved_data, 0, 0)
            #             gripper_pos = np.delete(gripper_pos, 0, 0)
            #             np.savez('./grasp/data/' + current_time + obj + '.npz',
            #                      loop_tac_data=tac_data,
            #                      all_tac_data=saved_data,
            #                      gripper_pos=gripper_pos)
            #         gripper.stop(gripper_index)
            #         gripper.fastStop(gripper_index)
            #
            #         # udp_socket.close()
            #         time.sleep(1)
            #         # gripper.stop(gripper_index)
            #         # time.sleep(1)
            #         gripper.disconnect()
            #         print('force stop exit')
            #         sys.exit(0)

            # if np.min(filted_data) < -1:
            #     gripper.stop(gripper_index)
            #     gripper.disconnect()
            #     print('force stop exit')
            #     sys.exit(0)
            # time.sleep(0.1)
            # print(tac_data.shape)
            # print('----')
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
            gripper.execute_command("skipbuffer")
            gripper.execute_command("abort")
            gripper.stop(gripper_index)
            gripper.fastStop(gripper_index)

            # udp_socket.close()
            time.sleep(1)
            # gripper.stop(gripper_index)
            # time.sleep(1)
            gripper.disconnect()
            gripper.close_socket()
            print('keyboard interrupt')
            sys.exit(0)


