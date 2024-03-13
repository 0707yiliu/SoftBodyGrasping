from schunk_gripper_v3 import SchunkGripper # integrate more functions
# from schunk_gripper import SchunkGripper
import time
import rtde_control
import rtde_receive

robot_ip = "10.42.0.162"
local_ip = '10.42.0.111'
# rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
# rtde_c = rtde_control.RTDEControlInterface(robot_ip)
gripper = SchunkGripper(local_ip=local_ip, local_port=44877)
gripper.connect(remote_function=True)
# rtde_c.servoJ(actual_q, 0.1, 0.1, 1, 0.2, 100) # for test rtdf with interpreter mode
gripper_index = 0
position1 = 10
position = 50
speed = 50
schunk_name = 'socket_grasp_sensor'
braking = 'true'
gripper.execute_command(f'EGUEGK_acknowledge("{gripper.rpc_socket_name}", 0)')
gripper.connect_server_socket()
gripper.execute_command(f'EGUEGK_setBrakingEnabled("{gripper.rpc_socket_name}", 0, "{braking}")')
gripper.execute_command(f'EGUEGK_brakeTest("{gripper.rpc_socket_name}", 0)')
# gripper.execute_command(f'EGUEGK_fastStop("{gripper.rpc_socket_name}", 0)')
gripper.getPosition(1)
for i in range(3):
    isDirectionOuter = "false"
    print('inside')
    gripper.execute_command(f'EGUEGK_simpleGrip("{gripper.rpc_socket_name}", 0, "{isDirectionOuter}", 1, 10)')
    time.sleep(1)
    gripper.execute_command(f'EGUEGK_getPosition()')
    gripper.execute_command(f'EGUEGK_stop("{gripper.rpc_socket_name}", 0)')
    isDirectionOuter = "true"
    print('outside')
    gripper.execute_command(f'EGUEGK_simpleGrip("{gripper.rpc_socket_name}", 0, "{isDirectionOuter}", 1, 10)')
    time.sleep(1)
    gripper.execute_command(f'EGUEGK_stop("{gripper.rpc_socket_name}", 0)')
    print('loop:', i)

gripper.disconnect()

'''
b'interpreter_mode()\n'
b'socket_open("10.42.0.162", 55050, "rpc_socket")\n'
b'socket_send_line("acknowledge(0)", "rpc_socket")\nsync()\n'
b'socket_close("rpc_socket")\n'
10.42.0.1 47013
b'socket_open("10.42.0.1", 47013, "socket_grasp_sensor")\n'
<socket.socket fd=6, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('10.42.0.1', 47013), raddr=('10.42.0.162', 47996)>
b'socket_open("127.0.0.1", 55050, "rpc_socket")\n'
b'socket_send_line("getPosition(0)", "rpc_socket")\n'
b'response=socket_read_line("rpc_socket", 2)\n'
b'popup(response)\n'
b'socket_send_line(response, "socket_grasp_sensor")\n
'''