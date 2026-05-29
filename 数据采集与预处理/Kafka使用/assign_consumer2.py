# 文件: assign_consumer2.py
from kafka import KafkaConsumer, TopicPartition
import time

consumer = KafkaConsumer(
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest'
)
consumer.assign([TopicPartition('assign_topic', 1)])

while True:
    msg = next(consumer)
    print("Partition 1 消息:", msg.value.decode('utf-8'))
