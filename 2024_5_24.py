class Book:
    def __init__(self, title, author, recommendation):
        self.title = title
        self.author = author
        self.recommendation = recommendation
        self.is_borrowed = False

    def __str__(self):
        status = "已借出" if self.is_borrowed else "未借出"
        return f"书名: {self.title}, 作家: {self.author}, 推荐语: {self.recommendation}, 状态: {status}"
class Library:
    def __init__(self):
        self.books = []

    def add_book(self, title, author, recommendation):
        book = Book(title, author, recommendation)
        self.books.append(book)
        print(f"书籍《{title}》已添加到系统。")

    def show_books(self):
        if not self.books:
            print("系统中没有书籍。")
        else:
            for book in self.books:
                print(book)

    def borrow_book(self, title):
        for book in self.books:
            if book.title == title:
                if book.is_borrowed:
                    print(f"书籍《{title}》已被借出。")
                else:
                    book.is_borrowed = True
                    print(f"书籍《{title}》已成功借出。")
                return
        print(f"未找到书籍《{title}》。")

    def return_book(self, title):
        for book in self.books:
            if book.title == title:
                if book.is_borrowed:
                    book.is_borrowed = False
                    print(f"书籍《{title}》已成功归还。")
                else:
                    print(f"书籍《{title}》未被借出。")
                return
        print(f"未找到书籍《{title}》。")


def main():
    library = Library()

    while True:
        print("\n图书管理系统")
        print("1. 查询书籍")
        print("2. 添加书籍")
        print("3. 借阅书籍")
        print("4. 归还书籍")
        print("5. 退出系统")

        choice = input("请选择操作（1-5）：")

        if choice == '1':
            library.show_books()
        elif choice == '2':
            title = input("请输入书名：")
            author = input("请输入作家：")
            recommendation = input("请输入推荐语：")
            library.add_book(title, author, recommendation)
        elif choice == '3':
            title = input("请输入要借阅的书名：")
            library.borrow_book(title)
        elif choice == '4':
            title = input("请输入要归还的书名：")
            library.return_book(title)
        elif choice == '5':
            print("退出系统。")
            break
        else:
            print("无效的选择，请重新输入。")


if __name__ == "__main__":
    main()
