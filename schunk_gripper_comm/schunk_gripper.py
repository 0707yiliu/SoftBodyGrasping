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
    EGUEGK_rpc_ip = "127.0.0.1"
    UR_INTERPRETER_SOCKET = 30020
    Enable_Interpreter_Socket = 30003
    STATE_REPLY_PATTERN = re.compile(r"(\w+):\W+(\d+)?") # DONOT USE IT NOW
    schunk_socket_name = "socket_grasp_sensor"
    gripper_index = 0
    ENCODING = 'UTF-8'
    EGUEGK_socket_uid = 0
    uid_th = 500

    def __init__(self):
        self.enable_interpreter_socket = None # the script port socket, for enabling interpreter mode
        self.socket = None # interpreter mode socket, for sending gripper command

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        # connect to the gripper's address
        # hostname: robot's ip
        # port: the local free port created for Schunk (not 30003 and 30020)
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.enable_interpreter_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((hostname, self.UR_INTERPRETER_SOCKET))
            self.socket.settimeout(socket_timeout)
            self.enable_interpreter_socket.connect((hostname, self.Enable_Interpreter_Socket))
            self.enable_interpreter_socket.settimeout(socket_timeout)
        except socket.error as exc:
            raise exc
        # enable interpreter mode
        try:
            self.scriptport_command("interpreter_mode()")
            time.sleep(4) # waiting the polyscope enable interpreter mode
        except:
            print("[ERROR] open interpreter mode failed")
            exit(0)
        # local socket open for Schunk
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {port}, "{self.schunk_socket_name}")')
            time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
    def disconnect(self) -> None:
        time.sleep(2) # waiting all of the command complete
        # self.clear()
        # time.sleep(2)
        self.execute_command(f'socket_close("{self.schunk_socket_name}")')
        time.sleep(2) # waiting robotic local socket close
        self.scriptport_command("end_interpreter()")
        self.socket.close()
        self.enable_interpreter_socket.close()

    def clear(self):
        self.scriptport_command("clear_interpreter()")

    def scriptport_command(self, command) -> None:
        # the port 30003 for urscript to communicate with UR robot
        if not command.endswith("\n"):
            command += "\n"
        # print(command.encode(self.ENCODING))
        self.enable_interpreter_socket.sendall(command.encode(self.ENCODING))

    def execute_command(self, command):
        # the port 30020 for interpreter mode to communicate with the binding port for Schunk
        if not command.endswith("\n"):
            command += "\n"
        # print(command.encode(self.ENCODING))
        self.socket.sendall(command.encode(self.ENCODING))
        data = self.socket.recv(1024)

    def get_reply(self):
        """
        read one line from the socket
        :return: text until new line
        """
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
        self.schunkHelperFunc(command)

    def moveRelative(self, gripperIndex, position, speed):
        command = "relative(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
        self.schunkHelperFunc(command)

    def grip(self, gripperIndex, isDirectionOuter, position, force, speed):
        command = "grip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
            position) + ", " + str(force) + ", " + str(speed) + ")"
        self.schunkHelperFunc(command)

    def simpleGrip(self, gripperIndex, isDirectionOuter, force, speed):
        command = "simpleGrip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
            force) + ", " + str(speed) + ")"
        self.schunkHelperFunc(command)

    def release(self, gripperIndex):
        command = "release(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(command)

    def fastStop(self, gripperIndex):
        command = "fastStop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(command)

    def stop(self, gripperIndex):
        command = "stop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(command)

    def acknowledge(self, gripperIndex):
        command = "acknowledge(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(command)
        time.sleep(0.5)

    def waitForComplete(self, gripperIndex, timeout = 10000):
        command = "waitForComplete(" + str(gripperIndex) + ", " + str(timeout) + ")"
        self.schunkHelperFunc(command)

    def setBrakingEnabled(self, gripperIndex, braking):
        command = "setBrakingEnabled(" + str(gripperIndex) + ", " + str(braking) + ")"
        self.schunkHelperFunc(command)

    def brakeTest(self, gripperIndex):
        command = "brakeTest(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(command)

    # Status Commands ----------------------------------------
    def status_rpcCall(self, socket_name, command):
        # open another name
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.port}, "{socket_name}")')
            time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)


    def getPosition(self, gripperIndex): # TODO: use other socket name, see egk_contribution.script
        command = "getPosition(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_position_" + str(self.EGUEGK_getNextId()))
        self.status_rpcCall(socket_name, command)
        # response = self.schunkHelperFunc(str("socket_status_position_" + str(EGUEGK_getNextId())), command)

    # Schunk Helper Functions --------------------------------
    def schunkHelperFunc(self, command):
        # send command
        # command += "\n\tsync()\n"
        self.execute_command(f'socket_send_line("{command}", "{self.schunk_socket_name}")\n')
