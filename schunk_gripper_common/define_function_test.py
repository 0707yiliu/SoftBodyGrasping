from schunk_gripper_v3 import SchunkGripper

HOST = "10.42.0.162" # ip
REMOTE = "10.42.0.1"
gripper_index = 0 # schunk id
local_ip = '10.42.0.111'
gripper = SchunkGripper(local_ip=local_ip, local_port=44607)
gripper.connect(remote_function=True)
# gripper.acknowledge(gripper_index)
#
# gripper.execute_command(f'def EGUEGK_rpcCall(socket_name, socket_address, socket_port, command, timeout = 2): socket_open(socket_address, socket_port, socket_name) socket_send_line(command, socket_name) sync() response = socket_read_line(socket_name, timeout) socket_close(socket_name) return response end')
# gripper.execute_command(f'def EGUEGK_executeCommand(socket_name, command, timeout = 1000): response = EGUEGK_rpcCall(socket_name, "{gripper.EGUEGK_rpc_ip}", {gripper.rpc_port}, command, timeout) return response end')
# gripper.execute_command(f'def EGUEGK_acknowledge(socket_name, gripperIndex): command = "acknowledge(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end')
gripper.execute_command(f'def foo1(val, a = 2): val = val + 2 + a return val end')
gripper.execute_command(f'def foo(val): val1 = foo1(val) popup(val1) end')

# gripper.execute_command(f'EGUEGK_acknowledge("{gripper.rpc_socket_name}", 0)')
gripper.execute_command(f'foo(52)')
gripper.disconnect()
# import socketserver
#
# class MyTCPHandler(socketserver.BaseRequestHandler):
#     """
#     The request handler class for our server.
#
#     It is instantiated once per connection to the server, and must
#     override the handle() method to implement communication to the
#     client.
#     """
#
#     def handle(self):
#         # self.request is the TCP socket connected to the client
#         self.data = self.request.recv(1024).strip()
#         print("Received from {}:".format(self.client_address[0]))
#         print(self.data)
#         # just send back the same data, but upper-cased
#         self.request.sendall(self.data.upper())
#
# if __name__ == "__main__":
#     HOST, PORT = "localhost", 9999
#
#     # Create the server, binding to localhost on port 9999
#     with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
#         # Activate the server; this will keep running until you
#         # interrupt the program with Ctrl-C
#         server.serve_forever()
