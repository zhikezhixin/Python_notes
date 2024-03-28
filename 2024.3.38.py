def menu() :
    print("----- 1.查询余额 ------")
    print("----- 2.存款     ------")
    print("----- 3.取款     ------")
    print("----- 0.退出      -----")
    print("--------------------")

money = 5000
name = input("请输入你的姓名：")

def show_money(rest_money) :
    print(f"{name}您好！您当前的余额是{rest_money}")

def increase_money(cash) :
    global money
    money+=cash
    print("存款成功！")
    show_money(money)

def get_money(cash) :
    global money
    money-=cash
    show_money(money)
while True :
    print("欢迎来到自助存取款一体机，请选择你的操作：")
    menu()
    Input = int(input(""))
    if Input == 1 :
        show_money(money)
    elif Input == 2 :
        cash = int(input("请输入您要存入的金额："))
        increase_money(cash)
    elif Input == 3 :
        cash = int(input("请输入您要取出的金额："))
        get_money(cash)
    elif Input == 0 :
        break
    else :
        print("输入错误，请重新输入！")
