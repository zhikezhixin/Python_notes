# 文件: commit_consumer.py
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'json_topic',
    bootstrap_servers='localhost:9092',
    group_id='test1',
    auto_offset_reset='earliest',
    enable_auto_commit=False
)

for message in consumer:
    data = json.loads(message.value.decode('utf-8'))
    print(data)
    consumer.commit()
