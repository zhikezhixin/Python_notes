# 文件: commit_producer.py
from kafka import KafkaProducer
import json

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
producer.send('json_topic', data)
producer.close()
print("data.json 数据已发送到 json_topic")
