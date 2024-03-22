# 输出任意自然大的数各个位数之和
# num = int(input("请输入一个任意大的自然数"))
# sum = 0
# while num%10 :
#     sum+=(num%10)
#     num//=10
# print(sum)
# 操作运算符
# a = int(input("请输入a的值"))
# b = int(input("请输入b的值"))
# print(f"a/b的结果是{a/b}")
# print(f"a//b的结果是{a//b}")
# print(f"a%b的结果是{a%b}")
# 猜数字游戏
import random

num = random.randint(1,100)
guess =  -1
count = 0
while  guess != num :
    guess = int(input("请输入你猜测的数字："))
    count+=1
    if guess > num :
        print("猜大了")
    elif guess == num :
        print(f"恭喜你第{count}次猜对了")
    else :
        print("猜小了")


