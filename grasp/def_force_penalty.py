# University Gent - imec 25/01/2024
# Auther: Yi Liu
# Description: Basic grasping function

class force_penalty:
    def __init__(self):
        self.init_force = 0

    def zero_force(self, force, cur_pos_finger, des_pos_finger):
    # keep the touching force zero for grasping.
    # INPUT: history force, current position of finger, desired position of finger
    # OUTPUT: next position of finger
        desired_force = 0
        diff_pos_finger = cur_pos_finger - des_pos_finger
        diff_force = force - desired_force
        if diff_force > 0: # the threshold force sensor
            next_pos_finger = -0.1 # next finger command
        else:
            next_pos_finger = 0
        return next_pos_finger
