import argparse
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener


class Visualiser:
    def __init__(self, topic_data_type, description):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument('--topic_name', type=str, default='MagTouch0', help='Name of the topic where the sensor data is published')
        args = self.parser.parse_args()
        self.topic_name = args.topic_name
        # self.topic_name = 'MagTouch0'
        # print(self.topic_name)
        listener = LivelinessListener(topic_name=self.topic_name)
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, self.topic_name, topic_data_type)
        self.reader = DataReader(domain_participant, topic, listener=listener)  # you can also supply Subscriber
        # instead of DomainParticipant to the DataReader constructor, but is not necessary

    def run(self):
        raise NotImplementedError
