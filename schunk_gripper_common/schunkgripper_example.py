from schunk_gripper import SchunkGripper
import time
# robot info
HOST = "10.42.0.162" # ip
PORT = 55050 # robotic local ip for schunk (RPC_PORT)
gripper_index = 0 # schunk id

position = 10 # test gripper position
position1 = 30
# position2 = 30
speed = 50 # test speed

directorOuter = False # control the gripper's moving direction

gripper = SchunkGripper() # create gripper
gripper.connect(HOST, PORT) # connect gripper
gripper.acknowledge(gripper_index) # active SchunkGripper

gripper.moveAbsolute(gripper_index, position, speed) # set position
time.sleep(2)

gripper.moveAbsolute(gripper_index, position1, speed)
time.sleep(2)
# gripper.moveAbsolute(gripper_index, position2, speed)
# time.sleep(0.5)
if directorOuter is True:
    dir_pos = -position * 0.7 # open finger
else:
    dir_pos = position * 0.7 # close finger
gripper.moveRelative(gripper_index, dir_pos, speed) # move based on current position, increment value
# gripper.waitForComplete(gripper_index)
time.sleep(2)
print("get position command...")
print(gripper.getPosition()) # TODO: the get_XXX statu-functions are wrong, fix it with egk_contribution.script
time.sleep(2)
if directorOuter is True:
    isDirOuter = "true"
else:
    isDirOuter = "false"
# gripper.grip(gripper_index, isDirOuter, position=80, force=1, speed=10)
# gripper.simpleGrip(gripper_index, isDirOuter, force=1, speed=10)

time.sleep(1)
gripper.disconnect() # disconnect gripper