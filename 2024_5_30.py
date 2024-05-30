#                           工作量计算器
# standard_hours = 80  # 一个标准项目需要的工时数量
# def calculate_work_hours(project_size, num_people):
#     work_hours = (project_size * standard_hours) / num_people
#     return work_hours
#
# def calculate_num_people(project_size, total_hours):
#     num_people = (project_size * standard_hours) / total_hours
#     return int(num_people) if num_people.is_integer() else int(num_people) + 1
# # 案例1：工时计算
# project_size = 1.5
# num_people = 2
# work_hours = calculate_work_hours(project_size, num_people)
# print(f"{project_size}倍标准大小的项目，一共有{num_people}人，需要{work_hours:.1f}个工时")
#
# # 案例2：人力计算
# project_size = 0.5
# total_hours = 20
# num_people = calculate_num_people(project_size, total_hours)
# print(f"{project_size}倍标准大小的项目，需要在{total_hours:.1f}个工时内完成，一共需要{num_people}个人力")


#                              抓狐狸游戏
# import random
# # 初始化狐狸位置
# fox_position = random.randint(0, 4)
#
# while True:
#     guess = int(input("请输入你猜测的洞口编号 (0-4): "))
#     if guess < 0 or guess > 4:
#         print("请输入0到4之间的数字。")
#         continue
#
#     if guess == fox_position:
#         print("恭喜你抓到了狐狸！")
#         break
#     else:
#         print("没抓到狐狸，狐狸跳到隔壁洞里去了。")
#         # 更新狐狸位置
#         if fox_position == 0:
#             fox_position = 1
#         elif fox_position == 4:
#             fox_position = 3
#         else:
#             fox_position += random.choice([-1, 1])


#                             用户注册信息管理系统
# import os
# def menu():
#     print("用户注册信息管理系统")
#     print("1. 显示全部已注册用户")
#     print("2. 查找/修改/删除用户信息")
#     print("3. 添加新用户")
#     print("4. 保存用户数据")
#     print("5. 退出系统")
#
# def load_users():
#     users = []
#     if os.path.exists("users.txt"):
#         with open("users.txt", "r") as f:
#             for line in f:
#                 username, password = line.strip().split(",")
#                 users.append((username, password))
#     return users
#
# def save_users(users):
#     with open("users.txt", "w") as f:
#         for username, password in users:
#             f.write(f"{username},{password}\n")
#
# def main():
#     users = load_users()
#     while True:
#         menu()
#         choice = input("请输入序号选择对应菜单: ")
#         if choice == "1":
#             print("当前已注册用户信息如下：")
#             for i, (username, password) in enumerate(users, 1):
#                 print(f"{i}. username={username}   password={password}")
#             input("按Enter键继续……")
#         elif choice == "2":
#             username = input("请输入要查找的用户名: ")
#             found = False
#             for i, (u, p) in enumerate(users):
#                 if u == username:
#                     found = True
#                     print(f"{username}已经注册！")
#                     operation = input("请选择操作：\n1. 修改用户\n2. 删除用户\n请输入序号选择对应操作: ")
#                     if operation == "1":
#                         new_username = input("请输入新的用户名: ")
#                         new_password = input("请输入新用户登录密码: ")
#                         users[i] = (new_username, new_password)
#                         print("已成功修改用户！")
#                     elif operation == "2":
#                         del users[i]
#                         print("已成功删除用户！")
#                     break
#             if not found:
#                 print(f"{username}不存在！")
#                 input("按Enter键继续……")
#         elif choice == "3":
#             username = input("请输入新的用户名: ")
#             password = input("请输入新用户登录密码: ")
#             users.append((username, password))
#             print("已成功添加用户！")
#         elif choice == "4":
#             save_users(users)
#             print("已成功保存用户信息！")
#         elif choice == "5":
#             print("谢谢使用，系统已退出！")
#             break
#         else:
#             print("无效的选择，请重新输入！")
#
# if __name__ == "__main__":
#     main()

#                                    流浪图书的旅途
# class Book:
#     def __init__(self, title, author, recommendation):
#         self.title = title
#         self.author = author
#         self.recommendation = recommendation
#         self.is_borrowed = False
#
#     def __str__(self):
#         status = "已借出" if self.is_borrowed else "未借出"
#         return f"书名: {self.title}, 作者: {self.author}, 推荐语: {self.recommendation}, 状态: {status}"
#
# class BookManager:
#     def __init__(self):
#         self.books = []
#
#     def menu(self):
#         while True:
#             print("\n图书管理系统")
#             print("1. 查询所有书籍")
#             print("2. 添加书籍")
#             print("3. 借阅书籍")
#             print("4. 归还书籍")
#             print("5. 退出系统")
#             choice = input("请输入数字选择对应的功能: ")
#
#             if choice == "1":
#                 self.show_all_books()
#             elif choice == "2":
#                 self.add_book()
#             elif choice == "3":
#                 self.lend_book()
#             elif choice == "4":
#                 self.return_book()
#             elif choice == "5":
#                 print("谢谢使用，系统已退出")
#                 break
#             else:
#                 print("无效的选择，请重新输入。")
#
#     def show_all_books(self):
#         if not self.books:
#             print("当前没有书籍。")
#         else:
#             print("当前所有书籍信息如下：")
#             for book in self.books:
#                 print(book)
#
#     def add_book(self):
#         title = input("请输入书名: ")
#         author = input("请输入作者: ")
#         recommendation = input("请输入推荐语: ")
#         new_book = Book(title, author, recommendation)
#         self.books.append(new_book)
#         print("书籍添加成功！")
#
#     def lend_book(self):
#         title = input("请输入要借阅的书名: ")
#         for book in self.books:
#             if book.title == title:
#                 if not book.is_borrowed:
#                     book.is_borrowed = True
#                     print(f"《{title}》借阅成功！")
#                     return
#                 else:
#                     print(f"《{title}》已被借出。")
#                     return
#         print(f"未找到书籍《{title}》。")
#
#     def return_book(self):
#         title = input("请输入要归还的书名: ")
#         for book in self.books:
#             if book.title == title:
#                 if book.is_borrowed:
#                     book.is_borrowed = False
#                     print(f"《{title}》归还成功！")
#                     return
#                 else:
#                     print(f"《{title}》未被借出。")
#                     return
#         print(f"未找到书籍《{title}》。")
#
# if __name__ == "__main__":
#     manager = BookManager()
#     manager.menu()

