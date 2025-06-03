 #列表
list_1 = ["this","is","a","list"]
print(f"列表中的内容是{list_1}")
print(f"列表的类型是{type(list_1)}")

list_2 = [[1,2,3],[4,5,6],[ 7,8,9]]
print(f"下标为[1][2]的元素是{list_2[1][2]}")
print(f"下标为[-1][-2]的元素是{list_2[-1][-2]}")

loc_1 = list_1.index("this")
print(f"列表中元素'this'的下标是{loc_1}")# 嵌套的列表无法查找下标

list_1[3] = "colledge"
print(f"修改后的list_1内容为{list_1}")

list_1.append("named") #在列表尾部追加元素
print(f"使用'append'追加后的列表list_1内容是{list_1}")

ext = ["hefei","universaty"]
list_1.extend(ext) #将其它数据容器取出，放置列表尾部
print(f"使用'extend'追加后的列表list_1的内容是{list_1}")

list_1.insert(0,"wellcome！") #插入元素 insert(下标，元素)
print(f"使用insert插入后列表list_1的内容是{list_1}")

del list_1[-2] # del直接删除列表中元素
print(f"使用del删除元素后列表list_1的内容是{list_1}")

ret = list_1.pop(0) #找到下标对应元素，删除后，返回被删除的元素
print(f"使用pop删除元素后列表list_1的内容是{list_1}")
print(f"被删除的元素是{ret}")

list_1.remove("a") #删除元素在列表中第一个匹配项
print(f"使用remove删除元素后列表list_1的内容是{list_1}")

tem = list_1.count("this") #统计某元素在列表中的数量
print(f"'this'在列表'list_1'中的数量是{tem}")

tmp = len(list_2)
print(f"'list_2'中元素的数量是{tem}")

list_1.clear()
print(f"列表被清空后{list_1}")

 #取出列表中偶数，并存到一个新的列表中
list = [1,2,3,4,5,6,7,8,9]
lis_1 = []
lis_2 = []
i = 0
while i < len(list) :
    if list[i]%2==0 :
        lis_1.append(list[i])
    i+=1
print(f"使用while循环结果是{lis_1}")

for x in list :
    if x%2 == 0 :
        lis_2.append(x)
print(f"使用for循环的结果是{lis_2}")


 #元组（与列表相似，但内容不可修改）
tpl = ("this","is","a","tuple")
tup_1 = tpl.index("this")
tup_2 = tpl.count("a")
tup_3 = len(tpl)
print(f"'this'的下标为{tup_1}")
print(f"'a'在元表tpl中出现的次数是{tup_2}")
print(f"元组的元素个数是{tup_3}")


 # 字符串 (与元组相同字符串也是一个无法修改的数字容器,
 #        因此对字符串修改分割等操作实际上都是复制到
 #        新的字符串中，对新字符串中内容进行改动，并返回)
Str = "wellcome to HFUU"
print(Str[-1])
print(Str[0:-1])
print("'l'在字符串中出现的次数为%d"%(Str.index("l")))

str_1 = "hello"
str_2 = Str.replace("HFUU",str_1)
print("将Str中'HFUU'换成'hello'的结果是%s"%(str_2))
print(Str)  #修改后返回新的字符串，原字符串不会变化

list_str = Str.split(" ")
print(f"分隔后的字符串是{list_str}")

Str_2 = "\tHello world"
str_3 = Str_2.strip("\t") # 移除首尾位子的字符；给不参数时，默认移除首尾位置空格
print(str_3)

print("U在字符串'Str'中出现的次数是%d"%(Str.count("U")))
print("字符串'Str'的字符个数时%d"%(len(Str)))