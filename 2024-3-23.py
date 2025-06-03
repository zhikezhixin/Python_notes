# 公司发礼物：1. 18~30的成年人
#           2. 同时入职时间大于两年或者级别大于三
age = int(input("请输入年龄："))
if age>18 and age<30 :
    work_time = int(input("请输入你的入职时间："))
    rank = int(input("请输入你的等级："))
    if work_time>2 :
        print("恭喜你，符合领取条件")
    elif rank>3 :
        print("恭喜你，符合领取条件")
    else :
        print("很抱歉，你的入职时间或等级不符合领取条件")
else :
    print("很抱歉，年龄不符合领取条件")


# 不换行功能
print("hello")
print("world") #python中print自带换行功能
print("hello",end='')
print("world",end='')


#水平制表符
print("How\tare\tyou")
print("I\tam\tfine")


# 测试for函数
name = "xiaomei"
for x in name :
    print(x,end='')
print()
for x in "xiaoming" :
    print(x, end='')

# 测试range函数--左边取不到
for x in range(8) :
    print(x, end='')
print()

for x in range(4,8) :
    print(x, end='')
print()

for x in range(1,8,3) :
    print(x, end='')
print()


# 输出九九乘法表
for i in range(1,10) :
    j = 1
    for j in range(1,i+1) :
        print(f"{j}*{i}={i*j}\t",end='')
    print()

