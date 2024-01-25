# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping

from detection_common.det_common import Det_Common
# object detection function
from schunk_gripper_common.schunk_gripper import SchunkGripper
# schunk gripper function
from ur_ikfast import ur_kinematics
# UR IK solver

det_com = Det_Common()
gripper = SchunkGripper()

# example of ur ikfast -------------
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
# -----------------------------------






