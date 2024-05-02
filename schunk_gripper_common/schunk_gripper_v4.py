import os.path

import pyschunk.tools.mylogger
from bkstools.bks_lib.bks_base import keep_communication_alive_input
from bkstools.bks_lib.bks_module import BKSModule, HandleWarningPrintOnly  # @UnusedImport
from bkstools.bks_lib.debug import Print, Var, ApplicationError, g_logmethod  # @UnusedImport


logger = pyschunk.tools.mylogger.getLogger( "BKSTools.demo.demo_grip_workpiece_with_position" )
pyschunk.tools.mylogger.setupLogging()
g_logmethod = logger.info

from bkstools.bks_lib import bks_options

class SchunkGripper:

    def __init__(self):
        prog = os.path.basename(globals()["__file__"])
        parser = bks_options.cBKSTools_OptionParser(prog=prog,
                                                    description=__doc__,  # @UndefinedVariable
                                                    additional_arguments=["grip_velocity",
                                                                          "force"])

        self.args = parser.parse_args()
        self.bks = BKSModule( self.args.host,
                     sleep_time=None,
                     #handle_warning=HandleWarningPrintOnly
                     debug=self.args.debug,
                     repeater_timeout=self.args.repeat_timeout,
                     repeater_nb_tries=self.args.repeat_nb_tries
                   )

    def connect(self) -> None:


    def close_socket(self):
        self.socket.close()
        self.enable_interpreter_socket.close()

    def _send_funtions(self):
        # !HelperFunction API
        EGUEGK_abs = f'def EGUEGK_abs(value): if (value < 0): return -value end return value end'
        EGUEGK_socket_uid = f'global EGUEGK_socket_uid = 0'
        EGUEGK_getNextId = f'def EGUEGK_getNextId(): enter_critical EGUEGK_socket_uid = (EGUEGK_socket_uid + 1) % 100 uid = EGUEGK_socket_uid exit_critical return uid end'
        EGUEGK_rpcCall = f'def EGUEGK_rpcCall(socket_name, socket_address, socket_port, command, timeout = 2): socket_open(socket_address, socket_port, socket_name) socket_send_line(command, socket_name) response = socket_read_line(socket_name, timeout) socket_close(socket_name) return response end'
        EGUEGK_executeCommand = f'def EGUEGK_executeCommand(socket_name, command, timeout = 1000): response = EGUEGK_rpcCall(socket_name, "{self.EGUEGK_rpc_ip}", {self.rpc_port}, command, timeout) return response end'
        EGUEGK_backInfo = f'def EGUEGK_backInfo(response, localhost, localport, socket_name): socket_open(localhost, localport, socket_name) socket_send_line(response, socket_name) socket_close(socket_name) end'
        # !Control Command API
        EGUEGK_moveAbsolute = f'def EGUEGK_moveAbsolute(socket_name, gripperIndex, position, speed): command = "absolute(" + to_str(gripperIndex) + ", " + to_str(position) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_moveRelative = f'def EGUEGK_moveRelative(socket_name, gripperIndex, position, speed): command = "relative(" + to_str(gripperIndex) + ", " + to_str(position) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_grip = f'def EGUEGK_grip(socket_name, gripperIndex, isDirectionOuter, position, force, speed): command = "grip(" + to_str(gripperIndex) + ", " + to_str(isDirectionOuter) + ", " + to_str(position) + ", " + to_str(force) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_release = f'def EGUEGK_release(socket_name, gripperIndex): command = "release(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_simpleGrip = f'def EGUEGK_simpleGrip(socket_name, gripperIndex, isDirectionOuter, force, speed): command = "simpleGrip(" + to_str(gripperIndex) + ", " + to_str(isDirectionOuter) + ", " + to_str(force) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_fastStop = f'def EGUEGK_fastStop(socket_name, gripperIndex): command = "fastStop(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_stop = f'def EGUEGK_stop(socket_name, gripperIndex): command = "stop(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_waitForComplete = f'def EGUEGK_waitForComplete(socket_name, gripperIndex, timeout = 10000): command = "waitForComplete(" + to_str(gripperIndex) + ", " + to_str(timeout) + ")" EGUEGK_executeCommand(socket_name + "waitForComplete", command, timeout + 1000) end'
        EGUEGK_setBrakingEnabled = f'def EGUEGK_setBrakingEnabled(socket_name, gripperIndex, braking): command = "setBrakingEnabled(" + to_str(gripperIndex) + ", " + to_str(braking) + ")" EGUEGK_executeCommand(socket_name + "braking", command) end'
        EGUEGK_brakeTest = f'def EGUEGK_brakeTest(socket_name, gripperIndex): command = "brakeTest(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_acknowledge = f'def EGUEGK_acknowledge(socket_name, gripperIndex): command = "acknowledge(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        # !Status Command API
        EGUEGK_getPosition = f'def EGUEGK_getPosition(gripperIndex): command = "getPosition(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_position_1"), command) EGUEGK_backInfo(response, "{self.localhost}", {self.localport}, "{self.schunk_socket_name}") end'
        EGUEGK_isCommandProcessed = f'def EGUEGK_isCommandProcessed(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isCommandProcessed(" + to_str(gripperIndex) + ")" response=EGUEGK_executeCommand(to_str("socket_status_cmd_processed_" + to_str(EGUEGK_getNextId())), command) response = EGUEGK_executeCommand(to_str("socket_status_cmd_processed_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isCommandReceived = f'def EGUEGK_isCommandReceived(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isCommandReceived(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_cmd_received_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_notFeasible = f'def EGUEGK_notFeasible(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isNotFeasible(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_not_feasible_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isReadyForOp = f'def EGUEGK_isReadyForOp(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isReadyForOperation(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_ready_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isPositionReached = f'EGUEGK_isPositionReached(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isPositionReached(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_position_reached_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isSWLimitReached = f'def EGUEGK_isSWLimitReached(gripperNumber = 1): gripperIndex = gripperNumber - 1  command = "isSoftwareLimitReached(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_software_limit_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isWorkpieceGripped = f'def EGUEGK_isWorkpieceGripped(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isWorkpieceGripped(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_workpiece_gripped_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isWrongWPGripped = f'def EGUEGK_isWrongWPGripped(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isWrongWorkpieceGripped(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_wrong_workpiece_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isNoWPDetected = f'def EGUEGK_isNoWPDetected(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isNoWorkpieceDetected(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_no_workpiece_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_isBrakeSet = f'def EGUEGK_isBrakeSet(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "isBrakeSet(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_brake_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_getError = f'def EGUEGK_getError(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "getError(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_error_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        EGUEGK_getWarning = f'def EGUEGK_getWarning(gripperNumber = 1): gripperIndex = gripperNumber - 1 command = "getWarning(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_warning_" + to_str(EGUEGK_getNextId())), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'

        UR_call = f'def UR_call(socket_name, socket_address, socket_port, command, timeout = 2): socket_open(socket_address, socket_port, socket_name) socket_send_line(command, socket_name) socket_close(socket_name) end'
        UR_ServoJ = f'def ServoJ(q, qd, qdd, time, lookahead_time, gain): popup(q) command = "servoj(" + to_str(q) + "," + to_str(qd) + "," + to_str(qdd) + "," + to_str(time) + "," + to_str(lookahead_time) + "," + to_str(gain))" UR_call("{self.ur_socket_name}", "{self.EGUEGK_rpc_ip}", {self.Enable_Interpreter_Socket}, command) end'
        all_list = [
                    # EGUEGK_abs, EGUEGK_socket_uid, EGUEGK_getNextId,
                    EGUEGK_rpcCall, EGUEGK_executeCommand, EGUEGK_backInfo,
                    EGUEGK_moveAbsolute, EGUEGK_moveRelative, EGUEGK_grip, EGUEGK_simpleGrip, EGUEGK_fastStop, EGUEGK_stop, EGUEGK_acknowledge,
                    EGUEGK_waitForComplete, EGUEGK_setBrakingEnabled, EGUEGK_brakeTest, EGUEGK_release,
                    EGUEGK_getPosition,
                    # UR_call, UR_ServoJ,
                    ]
        # EGUEGK_fastStop, EGUEGK_stop, EGUEGK_waitForComplete, EGUEGK_setBrakingEnabled, EGUEGK_brakeTest,
        # EGUEGK_getPosition
        for i in range(len(all_list)):
            self.execute_command(all_list[i])

    def disconnect(self) -> None:
        time.sleep(1)  # waiting all of the command complete
        self.scriptport_command("end_interpreter()")
        self.socket.close()
        self.enable_interpreter_socket.close()

    def clear(self):
        self.scriptport_command("clear_interpreter()")

    def connect_server_socket(self):
        # self.rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # self.rcv_socket.settimeout(5)
        # print(self.localhost, self.localport)
        # self.rcv_socket.bind((self.localhost, self.localport))
        # self.rcv_socket.listen()
        # command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
        # self.execute_command(command)
        # self.schunk_listener, self.schunk_addr = self.rcv_socket.accept()
        # print(self.schunk_listener)

        # !threading method
        self.t_schunk = threading.Thread(target=self._socket_server)
        self.t_schunk.setDaemon(True)
        self.t_schunk.start()

    def _socket_server(self):
        print("socket server buildup")
        rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rcv_socket.bind((self.localhost, self.localport))
        rcv_socket.listen()
        # command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
        # self.execute_command(command)

        while True:
            schunk_listener, schunk_addr = rcv_socket.accept()
            # print('get schunk listener:', schunk_listener)
            self.data = schunk_listener.recv(1024).decode(self.ENCODING)
            # print(self.data.decode(self.ENCODING))
            # print('recv data:', self.data)
            self.data = self.rpcGetResult(response=self.data)
            # print('typed data:', self.data)
            # try:
            #     schunk_listener, schunk_addr = rcv_socket.accept()
            #     # print('get schunk listener:', schunk_listener)
            #     self.data = schunk_listener.recv(1024).decode(self.ENCODING)
            #     # print(self.data.decode(self.ENCODING))
            #     # print('recv data:', self.data)
            #     self.data = self.rpcGetResult(response=self.data)
            #     print('typed data:', self.data)
            # except:
            #     print('accept the listener failed')
            #     sys.exit(0)

    def rpcGetResult(self, response):
        tokenIndex = str.find(response, ',')
        if tokenIndex < 0:
            tokenIndex = len(response)
        typeStr = response[:tokenIndex]
        resultStr = response[tokenIndex + 1:]
        if typeStr == 'boolean':
            resultStr = 'true'
            return resultStr
        elif typeStr == 'short' or typeStr == 'int' or typeStr == 'long':
            resultStr = int(resultStr)
            return resultStr
        elif typeStr == 'float' or typeStr == 'double':
            resultStr = float(resultStr)
            return resultStr

    def schunk_rpcCall(self, socket_name, command):
        # open another socket name for Schunk rpc_ip and port
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.rpc_port}, "{self.rpc_socket_name}")')
            # time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{self.rpc_socket_name}")\nsync()\n')
        self.execute_command(f'socket_close("{self.rpc_socket_name}")')

    def schunk_rpcCallRe(self, socket_name, command):
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.rpc_port}, "{socket_name}")')
            # command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
            # self.execute_command(command)
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{socket_name}")')
        self.execute_command(f'response=socket_read_line("{socket_name}", 2)')
        self.execute_command(f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")')
        self.execute_command(f'socket_send_line(response, "{self.schunk_socket_name}")')
        # self.execute_command(f'popup(response)\n')
        self.execute_command(f'socket_close("{socket_name}")')
        self.execute_command(f'socket_close("{self.schunk_socket_name}")')
        # print(self.schunk_listener)
        # self.data = self.schunk_listener.recv(1024)
        return self.data

    def scriptport_command(self, command) -> None:
        # the port 30003 for urscript to communicate with UR robot
        if not command.endswith("\n"):
            command += "\n"
        # print(command.encode(self.ENCODING))
        self.enable_interpreter_socket.sendall(command.encode(self.ENCODING))

    def servoJ(self, q, qd, qdd, _time, lookahead_time, gain):
        command = "servoj({}, {}, {}, {}, {}, {})".format(q, qd, qdd, _time, lookahead_time, gain)
        self.execute_command(command)

    def stopJ(self, a):
        command = "stopj({})".format(a)
        self.execute_command(command)

    def stopScript(self):
        command = "stopscript()"
        self.execute_command(command)

    def servoStop(self, a):
        command = "servostop({})".format(a)
        self.execute_command(command)

    def servoJ_inter(self, q, qd, qdd, _time, lookahead_time, gain):
        if self.remote_func is True:
            self.execute_command(f'ServoJ({q}, {qd}, {qdd}, {_time}, {lookahead_time}, {gain})')

    def execute_command(self, command):
        # the port 30020 for interpreter mode to communicate with the binding port for Schunk
        if not command.endswith("\n"):
            command += "\n"
        # print('sending command:', command.encode(self.ENCODING))
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
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_moveAbsolute("{self.rpc_socket_name}", {gripperIndex}, {position}, {speed})')
        else:
            command = "absolute(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def moveRelative(self, gripperIndex, position, speed):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_moveRelative("{self.rpc_socket_name}", {gripperIndex}, {position}, {speed})')
        else:
            command = "relative(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def grip(self, gripperIndex, isDirectionOuter, position, force, speed):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_grip("{self.rpc_socket_name}", {gripperIndex}, "{isDirectionOuter}", {position}, {force}, {speed})')
        else:
            command = "grip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
                position) + ", " + str(force) + ", " + str(speed) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def simpleGrip(self, gripperIndex, isDirectionOuter, force, speed):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_simpleGrip("{self.rpc_socket_name}", {gripperIndex}, "{isDirectionOuter}", {force}, {speed})')
        else:
            command = "simpleGrip(" + str(gripperIndex) + ", " + str(isDirectionOuter) + ", " + str(
                force) + ", " + str(speed) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def release(self, gripperIndex):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_release("{self.rpc_socket_name}", {gripperIndex})')
        else:
            command = "release(" + str(gripperIndex) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def fastStop(self, gripperIndex):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_fastStop("{self.rpc_socket_name}", {gripperIndex})')
        else:
            command = "fastStop(" + str(gripperIndex) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def stop(self, gripperIndex):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_stop("{self.rpc_socket_name}", {gripperIndex})')
        else:
            command = "stop(" + str(gripperIndex) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def acknowledge(self, gripperIndex):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_acknowledge("{self.rpc_socket_name}", {gripperIndex})')
        else:
            command = "acknowledge(" + str(gripperIndex) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)
            time.sleep(0.5)

    def waitForComplete(self, gripperIndex, timeout=10000):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_waitForComplete("{self.rpc_socket_name}", {gripperIndex}, {timeout})')
        else:
            command = "waitForComplete(" + str(gripperIndex) + ", " + str(timeout) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)
        # command = "waitForComplete(" + str(gripperIndex) + ", " + str(timeout) + ")"
        # self.schunkHelperFunc(self.schunk_socket_name, command)

    def setBrakingEnabled(self, gripperIndex, braking):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_setBrakingEnabled("{self.rpc_socket_name}", {gripperIndex}, "{braking}")')
        else:
            command = "setBrakingEnabled(" + str(gripperIndex) + ", " + str(braking) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    def brakeTest(self, gripperIndex):
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_brakeTest("{self.rpc_socket_name}", {gripperIndex})')
        else:
            command = "brakeTest(" + str(gripperIndex) + ")"
            self.schunkHelperFunc(self.schunk_socket_name, command)

    # Status Commands ----------------------------------------
    def getPosition(self, gripperNunber=1):  # TODO: use other socket name, see egk_contribution.script
        gripperIndex = gripperNunber - 1
        if self.remote_func is True:
            self.execute_command(f'EGUEGK_getPosition("{gripperIndex}")')
            return self.data
        else:
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
        return response

