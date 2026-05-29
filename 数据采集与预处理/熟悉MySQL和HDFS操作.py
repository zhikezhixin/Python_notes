import pyodbc

# 连接SQL Server数据库（使用Windows身份验证）
connect = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=ZEKER;'
    'DATABASE=DBTEST;'
    'Trusted_Connection=yes;'
)

# 获取游标
cursor = connect.cursor()

# 设置SQL语句（由于设置了外键级联删除，删除学生表记录会自动删除SC表相关记录）
sql = """
    DELETE FROM student 
    WHERE Sno = ?
"""

# 设置数据
data = ('10003',)

# 执行SQL语句
cursor.execute(sql, data)

# 提交事务
connect.commit()

# 输出执行结果
print("成功删除%d条记录" % cursor.rowcount)

# 关闭数据库连接
cursor.close()
connect.close()
