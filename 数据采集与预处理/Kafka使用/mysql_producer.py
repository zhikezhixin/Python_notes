# 文件: mysql_producer.py
from kafka import KafkaProducer
import json
import pymssql

# 连接 SQL Server
conn = pymssql.connect(
    server='ZEKER',
    user='Zeker',
    password='wh12138',  # 替换为实际密码
    database='school'
)
cursor = conn.cursor()
cursor.execute("SELECT sno, sname, ssex, sage FROM student;")
data = cursor.fetchall()
conn.close()

# Kafka 生产者
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for row in data:
    message = {
        'sno': row[0],
        'name': row[1],
        'sex': row[2],
        'age': row[3]
    }
    producer.send('mysql_topic', message)

producer.close()
print("数据已发送到 Kafka mysql_topic")
