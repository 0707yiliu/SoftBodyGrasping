import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicHermiteSpline
import math

def interpolate_position(start_pos, end_pos, num_points):
    # linear interpolation for position
    # output the operation space path points (position 3-D)
    delta_pos = end_pos - start_pos
    interval = delta_pos / num_points
    _pos = []
    for i in range(num_points):
        pos = start_pos + i * interval
        _pos.append(pos)
    return _pos

def interpolate_orientation(start_qua, end_qua, num_points):
    # linear interpolation for orientation (Quaternion)
    # output the operation space path points as quaternion (4-D)
    key_rot = Rotation.from_quat([start_qua, end_qua])
    key_time = [0, 1]
    _slerp = Slerp(key_time, key_rot)
    times = np.linspace(0, 1, num_points)
    _rot = _slerp(times).as_quat()
    return _rot

def joint_space_interpolate_quintic(
        start_angle, end_angle,
        time,
        start_vel, end_vel,
        start_acc, end_acc,
        num_points,
):
    # Input: start_angle: start joint angle, end_angle: end joint angle,
    #        time: the time you need for moving
    #        start_vel: start joint velocity, end_vel: end joint velocity
    #        start_acc: start joint acceleration, end_acc: end ...
    #        num_points: number of point between start and end
    # Output: the trajectory of joint angle, velocity and acceleration
    # tips: input is one item for the joint space interpolation but not a trajectory.
    assert start_angle.shape == end_angle.shape, 'angle length must be equal'
    # assert start_time.shape == end_time.shape, 'time length must be equal'
    assert start_vel.shape == end_vel.shape, 'velocity length must be equal'
    assert start_acc.shape == end_acc.shape, 'accelerate length must be equal'
    start_time = 0
    end_time = time
    _num_points = (end_time - start_time) / num_points
    q_list = []
    v_list = []
    a_list = []
    for i in range(len(start_angle)):
        # TODO: this loop can be replaced by a Matrix calculation
        a_0, a_1, a_2 = start_angle[i], start_vel[i], start_acc[i]
        a_3 = (20 * end_angle[i] - 20 * start_angle[i] - (8 * end_vel[i] + 12 * start_vel[i]) * end_time - (3 * end_acc[i] - start_acc[i]) * math.pow(end_time, 2)) / (2 * math.pow(end_time, 3))
        a_4 = (30 * start_angle[i] - 30 * end_angle[i] + (14 * end_vel[i] + 16 * start_vel[i]) * end_time + (3 * end_acc[i] - 2 * start_acc[i]) * math.pow(end_time, 2)) / (2 * math.pow(end_time, 4))
        a_5 = (12 * end_angle[i] - 12 * start_angle[i] - (6 * end_vel[i] + 6 * start_vel[i]) * end_time - (end_acc[i] - start_acc[i]) * math.pow(end_time, 2)) / (2 * math.pow(end_time, 5))

        _t = np.arange(start_time, end_time, _num_points)
        _q = a_0 + a_1 * _t + a_2 * np.power(_t, 2) + a_3 * np.power(_t, 3) + a_4 * np.power(_t, 4) + a_5 * np.power(_t, 5)
        _v = a_1 + 2 * a_2 * _t + 3 * a_3 * np.power(_t , 2) + 4 * a_4 * np.power(_t, 3) + 5 * a_5 * np.power(_t, 4)
        _a = 2 * a_2 + 6 * a_3 * _t + 12 * a_4 * np.power(_t, 2) + 20 * a_5 * np.power(_t, 3)

        q_list.append(_q)
        v_list.append(_v)
        a_list.append(_a)
    return q_list, v_list, a_list


