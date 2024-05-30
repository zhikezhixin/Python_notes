# import os
#
# class User:
#     def __init__(self, username, password):
#         self.username = username
#         self.password = password
#
#     def __str__(self):
#         return f"username={self.username}      password={self.password}"
#
# class UserRegistrationSystem:
#     def __init__(self):
#         self.users = []
#         self.load_users()
#
#     def load_users(self):
#         if os.path.exists("users.txt"):
#             with open("users.txt", "r") as file:
#                 for line in file:
#                     username, password = line.strip().split(",")
#                     self.users.append(User(username, password))
#
#     def save_users(self):
#         with open("users.txt", "w") as file:
#             for user in self.users:
#                 file.write(f"{user.username},{user.password}\n")
#
#     def display_all_users(self):
#         print("当前已注册用户信息如下：")
#         for i, user in enumerate(self.users, 1):
#             print(f"{i}. {user}")
#         input("按Enter键继续……")
#
#     def find_user(self, username):
#         for user in self.users:
#             if user.username == username:
#                 return user
#         return None
#
#     def add_user(self, username, password):
#         if self.find_user(username):
#             print("用户名已存在！")
#             return
#         self.users.append(User(username, password))
#         print("已成功添加用户")
#
#     def modify_user(self, username, new_username, new_password):
#         user = self.find_user(username)
#         if user:
#             user.username = new_username
#             user.password = new_password
#             print("已成功修改用户")
#         else:
#             print("用户不存在！")
#
#     def delete_user(self, username):
#         user = self.find_user(username)
#         if user:
#             self.users.remove(user)
#             print("已成功删除用户")
#         else:
#             print("用户不存在！")
#
#     def main_menu(self):
#         while True:
#             print("\n用户注册信息管理系统")
#             print("1．显示全部已注册用户")
#             print("2．查找/修改/删除用户信息")
#             print("3．添加新用户")
#             print("4．保存用户数据")
#             print("5．退出系统")
#             choice = input("请输入序号选择对应菜单: ")
#             if choice == "1":
#                 self.display_all_users()
#             elif choice == "2":
#                 self.search_modify_delete_user()
#             elif choice == "3":
#                 self.add_new_user()
#             elif choice == "4":
#                 self.save_users()
#                 print("已成功保存用户信息")
#                 input("按Enter键继续……")
#             elif choice == "5":
#                 print("谢谢使用，系统已退出")
#                 break
#             else:
#                 print("无效选择，请重新输入！")
#
#     def search_modify_delete_user(self):
#         username = input("请输入要查找的用户名: ")
#         user = self.find_user(username)
#         if user:
#             print(f"{username}已经注册！")
#             operation = input("请选择操作：\n1. 修改用户\n2．删除用户\n请输入序号选择对应操作:")
#             if operation == "1":
#                 new_username = input("请输入新的用户名: ")
#                 new_password = input("请输入新用户登录密码: ")
#                 self.modify_user(username, new_username, new_password)
#             elif operation == "2":
#                 self.delete_user(username)
#             else:
#                 print("无效选择，请重新输入！")
#         else:
#             print(f"{username}不存在！")
#         input("按Enter键继续……")
#
#     def add_new_user(self):
#         username = input("请输入新的用户名: ")
#         password = input("请输入新用户登录密码: ")
#         self.add_user(username, password)
#         input("按Enter键继续……")
#
# if __name__ == "__main__":
#     system = UserRegistrationSystem()
#     system.main_menu()


class Book:
    def __init__(self, title, author, recommendation):
        self.title = title
        self.author = author
        self.recommendation = recommendation
        self.borrowed = False  # 默认未借出

    def __str__(self):
        status = "已借出" if self.borrowed else "未借出"
        return f"书名：{self.title}\n作者：{self.author}\n推荐语：{self.recommendation}\n借阅状态：{status}"

    def borrow(self):
        if not self.borrowed:
            self.borrowed = True
            print("借阅成功！")
        else:
            print("该书已借出！")

    def return_book(self):
        if self.borrowed:
            self.borrowed = False
            print("归还成功！")
        else:
            print("该书未借出！")
class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)
        print("书籍添加成功！")

    def display_books(self):
        if not self.books:
            print("图书馆暂无书籍！")
        else:
            for index, book in enumerate(self.books, 1):
                print(f"书籍编号：{index}")
                print(book)
                print()

    def borrow_book(self, book_index):
        if 0 <= book_index < len(self.books):
            self.books[book_index].borrow()
        else:
            print("无效的书籍编号！")

    def return_book(self, book_index):
        if 0 <= book_index < len(self.books):
            self.books[book_index].return_book()
        else:
            print("无效的书籍编号！")
if __name__ == "__main__":
    # 创建几本书籍
    book1 = Book("Python编程入门", "张三", "学习Python的好书")
    book2 = Book("Java编程实践", "李四", "深入理解Java编程语言")
    book3 = Book("算法导论", "王五", "经典的算法学习教材")

    # 创建图书馆对象
    library = Library()

    # 将书籍添加到图书馆中
    library.add_book(book1)
    library.add_book(book2)
    library.add_book(book3)

    # 查询书籍
    print("图书馆所有书籍信息：")
    library.display_books()

    # 借阅书籍
    book_index = 0  # 假设借阅第一本书
    print(f"\n尝试借阅书籍 {book_index + 1}：")
    library.borrow_book(book_index)

    # 再次查询书籍
    print("\n借阅后图书馆所有书籍信息：")
    library.display_books()


    # 归还书籍
    print(f"\n尝试归还书籍 {book_index + 1}：")
    library.return_book(book_index)

    # 再次查询书籍
    print("\n归还后图书馆所有书籍信息：")
    library.display_books()
