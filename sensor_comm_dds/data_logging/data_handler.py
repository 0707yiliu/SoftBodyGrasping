import os
import csv
from loguru import logger
from datetime import *

import argparse
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader

from sensor_comm_dds.communication.data_classes.irtouch32 import IRTouch32
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener


class DataHandler:
    def __init__(self, topic_name, topic_data_type, directory, buffer_size=10):
        self.directory = directory
        self.topic_name = topic_name
        self.init_directory(self.directory)
        self.file_name = self.create_unique_file_name()
        self.csv_header = None
        self.buffer = []
        self.max_buffer_size = buffer_size
        self.current_data = None
        listener = LivelinessListener(topic_name=topic_name)
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, self.topic_name, topic_data_type)
        self.reader = DataReader(domain_participant, topic, listener=listener)  # you can also supply Subscriber
        # instead of DomainParticipant to the DataReader constructor, but is not necessary

    def create_unique_file_name(self, preamble=None, extension='.csv'):
        # Build the file name so that each experiment (a.k.a. each run of the code) saves data to a different file

        file_name = self.topic_name + "_" + str(date.today())
        if preamble:
            file_name = preamble + file_name
        file_number = 0
        for file_in_dir in os.listdir(self.directory):
            if file_in_dir.startswith(file_name):
                val = int(file_in_dir[file_in_dir.find("[") + 1:file_in_dir.find("]")])
                if val > file_number:
                    file_number = val
        file_name = file_name + "[" + str(file_number + 1) + "]" + extension

        return file_name

    def init_directory(self, directory):
        """
        Checks if directory exists, if not, creates it
        :param directory: directory to check
        """
        logger.warning(f'directory is {directory}')
        if not os.path.exists(directory):
            ans = input("Make new directory (" + directory + ")? [Y/N] ")
            if ans.lower() == "y":
                os.makedirs(directory)
            elif ans.lower() == "n":
                raise NotADirectoryError("Data directory doesn't exist, creation cancelled by user.")
            else:
                raise ValueError("Invalid answer given, enter [Y] or [N].")

    def init_csv_file(self, directory, file_name):
        if self.csv_header:
            with open(os.path.join(directory, file_name), 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                csv_writer.writerow(self.csv_header)

    def unpack_sample(self, sample):
        raise NotImplementedError

    def persist(self, data):
        """
        Persists a single readout from a single device
        :param data: list, the data
        :param device: string, the MAC address of the device
        """
        self.current_data = data
        self.buffer.append([datetime.now()] + data)

        if len(self.buffer) > self.max_buffer_size:
            self._persist_to_file()
            self.buffer = []

    def _persist_to_file(self):
        with open(os.path.join(self.directory, self.file_name), 'a+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';')
            for row in self.buffer:
                csv_writer.writerow(row)
        # logger.debug("Wrote to file")

    def read_single_sample(self):
        return self.unpack_sample(self.reader.read(N=1)[0])

    def run(self):
        for sample in self.reader.take_iter(timeout=None):
            self.persist(self.unpack_sample(sample))


class IRTouchDataHandler(DataHandler):
    def __init__(self, topic_name, directory, buffer_size=10, grid_size=(5, 4, 5, 4, 5, 4, 5)):
        super().__init__(topic_name=topic_name, topic_data_type=IRTouch32, directory=directory, buffer_size=buffer_size)
        num_taxels = sum(list(grid_size))
        self.csv_header = ['timestamp', 'strain'] + [f'taxel{i}' for i in range(num_taxels)]
        self.init_csv_file(self.directory, self.file_name)

    def unpack_sample(self, sample: IRTouch32):
        return [sample.strain_value] + list(sample.taxel_values)


class MagTouchDataHandler(DataHandler):
    def __init__(self, topic_name, directory="./data/smart_textile", buffer_size=10, grid_size=(2, 2)):
        super().__init__(topic_name=topic_name, topic_data_type=MagTouch4, directory=directory, buffer_size=buffer_size)
        grid_height = grid_size[0]
        grid_width = grid_size[1]
        self.num_taxels = grid_width * grid_height
        self.csv_header = ['PCB addr', 'timestamp']
        for i in range(self.num_taxels):
            self.csv_header += [f'taxel{i}_x', f'taxel{i}_y', f'taxel{i}_z']
        self.init_csv_file(self.directory, self.file_name)

    def unpack_sample(self, sample: MagTouch4):
        values = [0 for _ in range(self.num_taxels * 3)]
        for i, taxel in enumerate(sample.taxels):
            values[i * 3] = taxel.x
            values[i * 3 + 1] = taxel.y
            values[i * 3 + 2] = taxel.z
        return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This reader will log the data received on a topic to a .csv file.")
    parser.add_argument('topic_name', type=str, help='Name of the topic where the sensor data is published')
    parser.add_argument('sensor_type', type=str, help='Sensor type that is publishing to the topic: either "IRTOUCH32" or "MAGTOUCH"')
    args = parser.parse_args()
    topic_name = args.topic_name
    if args.sensor_type == "IRTOUCH32":
        data_handler = IRTouchDataHandler(topic_name=topic_name)
    elif args.sensor_type == "MAGTOUCH":
        data_handler = MagTouchDataHandler(topic_name=topic_name)
    else:
        raise ValueError("Invalid sensor type passed, check the help [-h] for this script")
    data_handler.run()
