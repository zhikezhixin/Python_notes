# 列表，元组，字符串统称序列，均支持切片操作
# 序列名[起始:结束:步长]
list = [1,2,3,4,5,6,7,8,9]
tup = ("a","b","c","d","e","f")
Str = "this is python"

print(list[9:1:-1]) # 步长为负数时，表示反向取
print(tup[1::2])
print(Str[::-1])

tem_1 = Str[8::] # 返回所需片段字符串，原字符串不变
print(tem_1)
print(f"对字符串进行切片操作后得到的类型是{type(tem_1)}")

tem_2 = list[1::]
print(tem_2)
print(f"对列表进行切片操作后得到的类型是{type(tem_2)}")


 # 练习从 "油加，给力奥，遍一天每" 中得到奥力给
str = "油加，给力奥，遍一天每"
tem_1 = str[-6:-9:-1]
print(f"法一：{tem_1}")
tem_2 = str.split("，")[1][::-1]
print(f"法二：{tem_2}")
tem_3 = str[3:6:1][::-1]
print(f"法三：{tem_3}")


#集合
set = {"learn","python","harder","python"}
print(f"set内容是{set}") #集合具有无序性，元素顺序随机

set.add("and")
print(f"增加元素后，集合set内容是{set}")

print(f"集合set元素个数是{len(set)}")

set.remove("harder")
print(f"删除'harder'元素后，集合set内容是{set}")

set.clear()
print(f"清空集合后{set}")

set_1 = {1,0,2,3}
set_2 = {5,3,2}
set_3 = set_1.difference(set_2) # 取出集合1中集合2没有的元素
                                # 集合1和集合2都不发生变化
set_3 = set_1.difference_update(set_2) # 消除集合1与集合2重复的元素
                                       # 集合1发生变化，集合2不变
set_4 = set.union(set_1,set_2) # 将集合2中元素整合到集合1中

print(f"取出1和2的差集后的结果是{set_3}")
print(f"取出1和2的差集后set_1的结果是{set_1}")
print(f"取出1和2的差集后set_2的结果是{set_2}")
print(f"消除差集后，集合1的结果是{set_1}")
print(f"消除差集后，集合2的结果是{set_2}")
print(f"合并集合后的结果是{set_4}")
print(f"合并集合后，集合1的结果是{set_1}")
print(f"合并集合后，集合2的结果是{set_2}")



