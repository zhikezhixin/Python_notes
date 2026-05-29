# 文件: mysql_consumer.py
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'mysql_topic',
    bootstrap_servers='localhost:9092',
    group_id=None,
    auto_offset_reset='earliest'
)

for msg in consumer:
    data = json.loads(msg.value.decode('utf-8'))
    print(data)
