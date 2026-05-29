# 文件: assign_producer.py
from kafka import KafkaProducer
import time
import uuid

producer = KafkaProducer(bootstrap_servers='localhost:9092')

while True:
    identifier = str(uuid.uuid4())
    producer.send('assign_topic', identifier.encode('utf-8'))
    time.sleep(1)  # 每秒发送一个
