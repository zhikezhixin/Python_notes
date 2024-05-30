import random

def display_holes(holes, fox_index):
    for i in range(len(holes)):
        if i == fox_index:
            print("[F]", end=" ")
        elif holes[i]:
            print("[ ]", end=" ")
        else:
            print("[X]", end=" ")
    print()

def main():
    holes = [False] * 5  # 初始化洞口，False表示洞口里没有狐狸
    fox_index = random.randint(0, 4)  # 随机选择一个洞口放置狐狸
    attempts = 1

    print("欢迎来到抓狐狸游戏！")

    while True:
        print("\n第{}次尝试：".format(attempts))
        display_holes(holes, fox_index)

        choice = int(input("请选择一个洞口（1-5）：")) - 1

        if choice < 0 or choice > 4:
            print("输入无效，请输入1到5之间的数字。")
            continue

        if choice == fox_index:
            print("恭喜！你成功抓到了狐狸！")
            break
        else:
            print("这个洞口没有狐狸，明天再来吧！")
            holes[choice] = False
            fox_index = (fox_index + 1) % 5  # 狐狸跳到隔壁洞口
            attempts += 1

    print("游戏结束，你一共尝试了{}次。".format(attempts))

if __name__ == "__main__":
    main()