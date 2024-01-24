from interpreter.interpreter import InterpreterHelper
import logging
import argparse
import sys
import time
import socket
import re
from socket_comm import send_to_socket

class SchunkGripper:
    # num of commands after which clear_interpreter() command will be invoked.
    # If interpreted statements are not cleared periodically then "runtime too much behind" error may
    # be shown when leaving interpreter mode
    CLEARBUFFER_LIMIT = 20 # DONOT USE IT NOW --- clear the buffer for interpreter mode (the number of command you want to perform)
    # EGUEGK_rpc_ip = "127.0.0.1"
    EGUEGK_rpc_ip = "10.42.0.162"
    UR_INTERPRETER_SOCKET = 30020
    Enable_Interpreter_Socket = 30003
    STATE_REPLY_PATTERN = re.compile(r"(\w+):\W+(\d+)?") # DONOT USE IT NOW
    schunk_socket_name = "socket_grasp_sensor"
    gripper_index = 0
    ENCODING = 'UTF-8'
    EGUEGK_socket_uid = 0
    uid_th = 500
    timeout = 1000

    def __init__(self):
        self.enable_interpreter_socket = None # the script port socket, for enabling interpreter mode
        self.socket = None # interpreter mode socket, for sending gripper command
        self.schunk_socket = None # schunk msg socket

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        # connect to the gripper's address
        # hostname: robot's ip
        # port: the local free port created for Schunk (not 30003 and 30020)
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.enable_interpreter_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.schunk_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((hostname, self.UR_INTERPRETER_SOCKET))
            self.socket.settimeout(socket_timeout)
            self.enable_interpreter_socket.connect((hostname, self.Enable_Interpreter_Socket))
            self.enable_interpreter_socket.settimeout(socket_timeout)
            self.schunk_socket.connect((hostname, self.port))
            self.schunk_socket.settimeout(socket_timeout)
        except socket.error as exc:
            raise exc
        # enable interpreter mode
        try:
            self.scriptport_command("interpreter_mode()")
            time.sleep(4) # waiting the polyscope enable interpreter mode
        except:
            print("[ERROR] open interpreter mode failed")
            exit(0)
        # # local socket open for Schunk
        # try:
        #     self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {port}, "{self.schunk_socket_name}")')
        #     time.sleep(2) # waiting the socket opened in robotic local network
        # except:
        #     print("[ERROR] open robotic local socket failed")
        #     exit(0)
    def disconnect(self) -> None:
        time.sleep(1) # waiting all of the command complete
        self.scriptport_command("end_interpreter()")
        self.socket.close()
        self.enable_interpreter_socket.close()

    def clear(self):
        self.scriptport_command("clear_interpreter()")

    def schunk_rpcCall(self, socket_name, command):
        # open another socket name for Schunk rpc_ip and port
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.port}, "{socket_name}")')
            # time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{socket_name}")\nsync()\n')
        self.execute_command(f'socket_close("{socket_name}")')

    def schunk_rpcCallRe(self, socket_name, command):
        # open another socket name for Schunk rpc_ip and port
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.port}, "{socket_name}")')
            # time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{socket_name}")\nsync()\n')
        #----schunk test---------
        # tmp_command = f'socket_send_line("{command}", "{socket_name}")\nsync()\n'
        # self.schunk_socket.sendall(tmp_command.encode(self.ENCODING))
        # tmp_command = f'socket_read_line("{socket_name}", {self.timeout})\n'
        # self.schunk_socket.sendall(tmp_command.encode(self.ENCODING))
        # response = self.schunk_socket.recv(1024)
        # print("response info from SCHUNK port:", response.decode(self.ENCODING))
        #------------------------
        self.execute_command(f'socket_read_line("{socket_name}", {self.timeout})\n')
        response = self.schunk_socket.recv(1024)
        self.execute_command(f'socket_close("{socket_name}")')
        return response
        # send get_XXX command

    def scriptport_command(self, command) -> None:
        # the port 30003 for urscript to communicate with UR robot
        if not command.endswith("\n"):
            command += "\n"
        print(command.encode(self.ENCODING))
        self.enable_interpreter_socket.sendall(command.encode(self.ENCODING))

    def execute_command(self, command):
        # the port 30020 for interpreter mode to communicate with the binding port for Schunk
        if not command.endswith("\n"):
            command += "\n"
        print(command.encode(self.ENCODING))
        self.socket.sendall(command.encode(self.ENCODING))
        # data = self.socket.recv(1024)
        # return data

    def get_reply(self):
        collected = b''
        while True:
            part = self.socket.recv(1)
            if part != b"\n":
                collected += part
            elif part == b"\n":
                break
        return collected.decode(self.ENCODING)

    def EGUEGK_getNextId(self):
        self.EGUEGK_socket_uid = (self.EGUEGK_socket_uid + 1) % self.uid_th
        # uid = self.EGUEGK_socket_uid
        return self.EGUEGK_socket_uid
        
    # Control API Commands ----------------------------------------
    def moveAbsolute(self, gripperIndex, position, speed):
        command = "absolute(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def moveRelative(self, gripperIndex, position, speed):
        command = "relative(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def grip(self, gripperIndex, isDirectionOuter, position, force, speed):
        command = "grip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
            position) + ", " + str(force) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def simpleGrip(self, gripperIndex, isDirectionOuter, force, speed):
        command = "simpleGrip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
            force) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def release(self, gripperIndex):
        command = "release(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def fastStop(self, gripperIndex):
        command = "fastStop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def stop(self, gripperIndex):
        command = "stop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def acknowledge(self, gripperIndex):
        command = "acknowledge(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)
        time.sleep(0.5)

    def waitForComplete(self, gripperIndex, timeout=10000):
        command = "waitForComplete(" + str(gripperIndex) + ", " + str(timeout) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def setBrakingEnabled(self, gripperIndex, braking):
        command = "setBrakingEnabled(" + str(gripperIndex) + ", " + str(braking) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    def brakeTest(self, gripperIndex):
        command = "brakeTest(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name,command)

    # Status Commands ----------------------------------------
    def getPosition(self, gripperNunber=1): # TODO: use other socket name, see egk_contribution.script
        gripperIndex = gripperNunber - 1
        command = "getPosition(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_position_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        # self.status_rpcCall(socket_name, command)
        # response = self.schunkHelperFunc(str("socket_status_position_" + str(EGUEGK_getNextId())), command)
        return response

    def isCommandProcessed(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isCommandProcessed(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_cmd_processed_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isCommandReceived(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isNotFeasible(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_cmd_received_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def notFeasible(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isNotFeasible(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_not_feasible_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isReadyForOp(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isReadyForOperation(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_ready_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isPositionReached(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isPositionReached(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_position_reached_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isSWLimitReached(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isSoftwareLimitReached(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_software_limit_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def getError(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "getError(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_error_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def getWarning(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "getWarning(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_warning_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    # Schunk Helper Functions --------------------------------
    def schunkHelperFunc(self, socket_name, command):
        # send command
        # command += "\n\tsync()\n"
        self.schunk_rpcCall(socket_name, command)

    def schunkHelperFuncRe(self, socket_name, command):
        # send command
        # command += "\n\tsync()\n"
        response = self.schunk_rpcCallRe(socket_name, command)
        # self.execute_command(f'socket_send_line("{command}", "{self.schunk_socket_name}")\n')
        return response.decode()

