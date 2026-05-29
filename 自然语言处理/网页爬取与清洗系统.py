

import re
import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime


class WebCrawler:
    """网页爬取核心类"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def fetch_html(self, url):
        """获取网页HTML内容"""
        try:
            # 添加更多重试和错误处理
            session = requests.Session()
            session.headers.update(self.headers)

            response = session.get(url, timeout=15, allow_redirects=True)

            # 尝试检测正确的编码
            if response.encoding == 'ISO-8859-1':
                encodings = requests.utils.get_encodings_from_content(response.text)
                if encodings:
                    response.encoding = encodings[0]
                else:
                    response.encoding = response.apparent_encoding

            return response.text
        except requests.exceptions.RequestException as e:
            raise Exception(f"网页获取失败: {str(e)}")

    def extract_text_from_html(self, html):
        """从HTML中提取文本内容 - 使用BeautifulSoup改进版"""
        try:
            # 使用BeautifulSoup解析
            soup = BeautifulSoup(html, 'html.parser')

            # 移除script、style、meta等标签
            for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'iframe']):
                tag.decompose()

            # 获取文本
            text = soup.get_text(separator=' ', strip=True)

            # 如果文本太短，尝试提取特定区域
            if len(text) < 100:
                # 尝试提取body内容
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)

            # 清理多余空白
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        except Exception as e:
            # 如果BeautifulSoup失败，使用正则表达式
            return self._extract_text_regex(html)

    def _extract_text_regex(self, html):
        """备用方案：使用正则表达式提取文本"""
        # 移除script和style标签及内容
        text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
        text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.IGNORECASE)

        # 移除HTML注释
        text = re.sub(r'<!--[\s\S]*?-->', '', text)

        # 移除所有HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)

        # 解码HTML实体
        html_entities = {
            '&nbsp;': ' ', '&lt;': '<', '&gt;': '>', '&amp;': '&',
            '&quot;': '"', '&#39;': "'", '&ldquo;': '"', '&rdquo;': '"',
            '&mdash;': '—', '&ndash;': '–', '&hellip;': '…'
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        # 清理空白
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def clean_text(self, text):
        """清洗文本内容"""
        if not text:
            return ""

        # 移除多余空白字符
        cleaned = re.sub(r'\s+', ' ', text)

        # 移除网址
        cleaned = re.sub(r'https?://[^\s]+', '', cleaned)
        cleaned = re.sub(r'www\.[^\s]+', '', cleaned)

        # 移除邮箱
        cleaned = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', cleaned)

        # 移除电话号码
        cleaned = re.sub(r'\d{3}[-\s]?\d{4}[-\s]?\d{4}', '', cleaned)
        cleaned = re.sub(r'\d{11}', '', cleaned)

        # 移除特殊符号(保留中文、英文、数字和基本标点)
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,，。.!！?？;；:：、\s()（）\[\]【】]', '', cleaned)

        # 移除连续标点
        cleaned = re.sub(r'([,，。.!！?？;；:：、]){2,}', r'\1', cleaned)

        # 移除单独的数字和字母
        cleaned = re.sub(r'\s+[a-zA-Z0-9]\s+', ' ', cleaned)

        # 去除首尾空白和标点
        cleaned = cleaned.strip().strip(',，。.!！?？;；:：、')

        return cleaned

    def save_to_file(self, text, filename):
        """保存文本到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            raise Exception(f"文件保存失败: {str(e)}")


class WebCrawlerGUI:
    """图形界面类"""

    def __init__(self, root):
        self.root = root
        self.root.title("网页文本爬取与清洗系统")
        self.root.geometry("1200x800")

        self.crawler = WebCrawler()
        self.raw_text = ""
        self.cleaned_text = ""

        self.setup_ui()

        # 添加推荐网址列表
        self.recommended_urls = [
            "https://baike.baidu.com/item/人工智能",
            "https://baike.baidu.com/item/机器学习",
            "https://baike.baidu.com/item/深度学习",
            "https://baike.baidu.com/item/自然语言处理",
        ]

    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        title_frame = tk.Frame(self.root, bg='#4F46E5', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="网页文本爬取与清洗系统",
                 font=('Arial', 20, 'bold'), bg='#4F46E5', fg='white').pack(pady=10)
        tk.Label(title_frame, text="自然语言处理实验 - 静态网页爬取与文本处理",
                 font=('Arial', 10), bg='#4F46E5', fg='white').pack()

        # 控制面板
        control_frame = tk.Frame(self.root, bg='white', padx=20, pady=15)
        control_frame.pack(fill='x', padx=10, pady=10)

        # URL输入框架
        url_frame = tk.Frame(control_frame, bg='white')
        url_frame.pack(fill='x', pady=5)

        tk.Label(url_frame, text="目标网页URL:", font=('Arial', 10), bg='white').pack(side='left', padx=5)

        self.url_entry = tk.Entry(url_frame, width=50, font=('Arial', 10))
        self.url_entry.pack(side='left', padx=5, fill='x', expand=True)
        self.url_entry.insert(0, "https://baike.baidu.com/item/人工智能")

        # 快速选择下拉框
        tk.Label(url_frame, text="快速选择:", font=('Arial', 9), bg='white').pack(side='left', padx=5)
        self.url_combo = ttk.Combobox(url_frame, width=20, font=('Arial', 9))
        self.url_combo['values'] = [
            "百度百科-人工智能",
            "百度百科-机器学习",
            "百度百科-Deep Learning",
            "百度百科-自然语言处理"
        ]
        self.url_combo.pack(side='left', padx=5)
        self.url_combo.bind('<<ComboboxSelected>>', self.on_url_select)

        # 按钮框架
        btn_frame = tk.Frame(control_frame, bg='white')
        btn_frame.pack(fill='x', pady=10)

        self.crawl_btn = tk.Button(btn_frame, text="🚀 开始爬取", command=self.start_crawl,
                                   bg='#4F46E5', fg='white', font=('Arial', 10, 'bold'),
                                   padx=20, pady=5, cursor='hand2')
        self.crawl_btn.pack(side='left', padx=5)

        tk.Button(btn_frame, text="🗑️ 清空", command=self.clear_results,
                  bg='#EF4444', fg='white', font=('Arial', 10, 'bold'),
                  padx=20, pady=5, cursor='hand2').pack(side='left', padx=5)

        tk.Button(btn_frame, text="📝 使用演示数据", command=self.use_demo_data,
                  bg='#10B981', fg='white', font=('Arial', 10, 'bold'),
                  padx=20, pady=5, cursor='hand2').pack(side='left', padx=5)

        # 统计信息框架
        self.stats_frame = tk.Frame(self.root, bg='white')
        self.stats_frame.pack(fill='x', padx=10, pady=5)

        self.stat_raw = self.create_stat_box(self.stats_frame, "原始字符数", "0", '#4F46E5')
        self.stat_cleaned = self.create_stat_box(self.stats_frame, "清洗后字符数", "0", '#10B981')
        self.stat_removed = self.create_stat_box(self.stats_frame, "移除字符数", "0", '#EF4444')

        # 结果展示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧 - 原始文本
        left_frame = tk.Frame(result_frame, bg='white', relief='solid', borderwidth=1)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)

        left_header = tk.Frame(left_frame, bg='#EEF2FF', height=40)
        left_header.pack(fill='x')
        left_header.pack_propagate(False)

        tk.Label(left_header, text="📄 原始爬取文本", font=('Arial', 12, 'bold'),
                 bg='#EEF2FF', fg='#4F46E5').pack(side='left', padx=10, pady=10)

        tk.Button(left_header, text="💾 下载", command=lambda: self.download_text(self.raw_text, 'raw_text.txt'),
                  bg='#C7D2FE', fg='#4F46E5', font=('Arial', 9), cursor='hand2').pack(side='right', padx=10)

        self.raw_text_box = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=('Microsoft YaHei', 10),
                                                      bg='#F9FAFB', relief='flat')
        self.raw_text_box.pack(fill='both', expand=True, padx=5, pady=5)

        # 右侧 - 清洗后文本
        right_frame = tk.Frame(result_frame, bg='white', relief='solid', borderwidth=1)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)

        right_header = tk.Frame(right_frame, bg='#ECFDF5', height=40)
        right_header.pack(fill='x')
        right_header.pack_propagate(False)

        tk.Label(right_header, text="✨ 清洗后文本", font=('Arial', 12, 'bold'),
                 bg='#ECFDF5', fg='#10B981').pack(side='left', padx=10, pady=10)

        tk.Button(right_header, text="💾 下载",
                  command=lambda: self.download_text(self.cleaned_text, 'cleaned_text.txt'),
                  bg='#A7F3D0', fg='#10B981', font=('Arial', 9), cursor='hand2').pack(side='right', padx=10)

        self.cleaned_text_box = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=('Microsoft YaHei', 10),
                                                          bg='#F0FDF4', relief='flat')
        self.cleaned_text_box.pack(fill='both', expand=True, padx=5, pady=5)

        # 底部说明
        info_frame = tk.Frame(self.root, bg='#DBEAFE', relief='solid', borderwidth=1)
        info_frame.pack(fill='x', padx=10, pady=5)

        info_text = """💡 使用说明: 输入网页URL后点击"开始爬取"，系统将自动提取并清洗文本 | 推荐使用百度百科等静态网页 | 支持文本下载保存 | 如遇问题可使用"演示数据"查看效果"""
        tk.Label(info_frame, text=info_text, font=('Arial', 9), bg='#DBEAFE',
                 fg='#1E40AF', wraplength=1100, justify='left').pack(padx=10, pady=8)

    def create_stat_box(self, parent, label, value, color):
        """创建统计信息框"""
        frame = tk.Frame(parent, bg='white', relief='solid', borderwidth=1)
        frame.pack(side='left', fill='x', expand=True, padx=5, pady=5)

        tk.Label(frame, text=label, font=('Arial', 9), bg='white', fg='#6B7280').pack(pady=(10, 5))
        value_label = tk.Label(frame, text=value, font=('Arial', 24, 'bold'), bg='white', fg=color)
        value_label.pack(pady=(0, 10))

        return value_label

    def on_url_select(self, event):
        """快速选择URL"""
        selection = self.url_combo.current()
        if selection >= 0 and selection < len(self.recommended_urls):
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, self.recommended_urls[selection])

    def use_demo_data(self):
        """使用演示数据"""
        demo_html = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head><title>人工智能技术</title></head>
        <body>
            <h1>人工智能的发展历程</h1>
            <p>人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
            它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。</p>

            <h2>发展阶段</h2>
            <p>人工智能从1956年诞生以来，经历了多个发展阶段：</p>
            <ul>
                <li>1956-1974年：黄金时代，充满乐观情绪</li>
                <li>1974-1980年：第一次AI寒冬</li>
                <li>1980-1987年：专家系统的繁荣</li>
                <li>1987-1993年：第二次AI寒冬</li>
                <li>1993-2011年：智能代理的崛起</li>
                <li>2011年至今：深度学习革命</li>
            </ul>

            <h2>关键技术</h2>
            <p>现代人工智能的核心技术包括：</p>
            <ol>
                <li>机器学习（Machine Learning）：让计算机能够从数据中学习</li>
                <li>Deep Learning（Deep Learning）：基于神经网络的学习方法</li>
                <li>自然语言处理（NLP）：让计算机理解和生成人类语言</li>
                <li>计算机视觉（Computer Vision）：让计算机"看懂"图像和视频</li>
                <li>强化学习（Reinforcement Learning）：通过试错来学习最优策略</li>
            </ol>

            <h2>应用领域</h2>
            <p>人工智能已经广泛应用于各个领域：医疗诊断、金融分析、自动驾驶、
            智能客服、推荐系统、语音识别、图像识别等。特别是在2023年，
            大语言模型如ChatGPT的出现，更是将AI技术推向了新的高度。</p>

            <div class="contact">
                <p>联系方式：Email: ai@example.com | 电话：010-12345678</p>
                <p>网址：https://www.ai-research.com</p>
            </div>

            <footer>
                <p>© 2025 AI研究中心 版权所有 [All Rights Reserved]</p>
                <p>访问量：123456@#$% 更新时间：2025-10-28!!!</p>
            </footer>
        </body>
        </html>
        """

        try:
            # 提取文本
            self.raw_text = self.crawler.extract_text_from_html(demo_html)
            self.raw_text_box.delete('1.0', tk.END)
            self.raw_text_box.insert('1.0', self.raw_text)

            # 清洗文本
            self.cleaned_text = self.crawler.clean_text(self.raw_text)
            self.cleaned_text_box.delete('1.0', tk.END)
            self.cleaned_text_box.insert('1.0', self.cleaned_text)

            # 更新统计
            raw_len = len(self.raw_text)
            cleaned_len = len(self.cleaned_text)
            removed_len = raw_len - cleaned_len

            self.stat_raw.config(text=str(raw_len))
            self.stat_cleaned.config(text=str(cleaned_len))
            self.stat_removed.config(text=str(removed_len))

            messagebox.showinfo("成功", "演示数据加载成功！")

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def start_crawl(self):
        """开始爬取"""
        url = self.url_entry.get().strip()

        if not url:
            messagebox.showwarning("警告", "请输入网页URL")
            return

        if not url.startswith(('http://', 'https://')):
            messagebox.showwarning("警告", "URL必须以 http:// 或 https:// 开头")
            return

        # 禁用按钮
        self.crawl_btn.config(state='disabled', text='⏳ 处理中...')
        self.root.update()

        try:
            # 获取HTML
            html = self.crawler.fetch_html(url)

            if not html or len(html) < 100:
                raise Exception("获取的网页内容过短，可能是网页加载失败或需要JavaScript渲染")

            # 提取文本
            self.raw_text = self.crawler.extract_text_from_html(html)

            if len(self.raw_text) < 50:
                raise Exception(
                    "提取的文本内容过少，建议：\n1. 尝试百度百科等静态网页\n2. 使用'演示数据'查看功能\n3. 检查网页是否需要JavaScript渲染")

            self.raw_text_box.delete('1.0', tk.END)
            self.raw_text_box.insert('1.0', self.raw_text)

            # 清洗文本
            self.cleaned_text = self.crawler.clean_text(self.raw_text)
            self.cleaned_text_box.delete('1.0', tk.END)
            self.cleaned_text_box.insert('1.0', self.cleaned_text)

            # 更新统计
            raw_len = len(self.raw_text)
            cleaned_len = len(self.cleaned_text)
            removed_len = raw_len - cleaned_len

            self.stat_raw.config(text=str(raw_len))
            self.stat_cleaned.config(text=str(cleaned_len))
            self.stat_removed.config(text=str(removed_len))

            messagebox.showinfo("成功",
                                f"爬取完成！\n\n原始字符: {raw_len}\n清洗后字符: {cleaned_len}\n移除字符: {removed_len}")

        except Exception as e:
            messagebox.showerror("错误",
                                 f"爬取失败！\n\n{str(e)}\n\n建议：\n• 使用百度百科等静态网页\n• 点击'使用演示数据'查看功能")

        finally:
            # 恢复按钮
            self.crawl_btn.config(state='normal', text='🚀 开始爬取')

    def clear_results(self):
        """清空结果"""
        self.raw_text = ""
        self.cleaned_text = ""
        self.raw_text_box.delete('1.0', tk.END)
        self.cleaned_text_box.delete('1.0', tk.END)
        self.stat_raw.config(text='0')
        self.stat_cleaned.config(text='0')
        self.stat_removed.config(text='0')

    def download_text(self, text, default_filename):
        """下载文本"""
        if not text:
            messagebox.showwarning("警告", "没有可下载的内容")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if filename:
            try:
                self.crawler.save_to_file(text, filename)
                messagebox.showinfo("成功", f"文件已保存至:\n{filename}")
            except Exception as e:
                messagebox.showerror("错误", str(e))


def main_gui():
    """启动图形界面"""
    root = tk.Tk()
    app = WebCrawlerGUI(root)
    root.mainloop()


def main_cli():
    """命令行版本"""
    print("=" * 70)
    print(" " * 20 + "网页文本爬取与清洗系统")
    print(" " * 25 + "命令行版本")
    print("=" * 70)

    crawler = WebCrawler()

    # 显示推荐URL
    print("\n推荐网址（复制使用）：")
    print("1. https://baike.baidu.com/item/人工智能")
    print("2. https://baike.baidu.com/item/机器学习")
    print("3. https://baike.baidu.com/item/深度学习")

    # 输入URL
    url = input("\n请输入目标网页URL（直接回车使用默认）: ").strip()
    if not url:
        url = "https://baike.baidu.com/item/人工智能"
        print(f"→ 使用默认URL: {url}")

    try:
        print("\n" + "=" * 70)
        print("[1/4] 正在获取网页内容...")
        html = crawler.fetch_html(url)
        print(f"✓ 网页获取成功 (HTML长度: {len(html)})")

        print("\n[2/4] 正在提取文本...")
        raw_text = crawler.extract_text_from_html(html)

        if len(raw_text) < 50:
            print("⚠ 警告：提取的文本过少，可能是动态网页")
            print("建议使用百度百科等静态网页")

        print(f"✓ 文本提取成功 (字符数: {len(raw_text)})")

        print("\n[3/4] 正在清洗文本...")
        cleaned_text = crawler.clean_text(raw_text)
        print(f"✓ 文本清洗完成 (字符数: {len(cleaned_text)})")

        print("\n[4/4] 正在保存文件...")
        crawler.save_to_file(raw_text, 'raw_text.txt')
        crawler.save_to_file(cleaned_text, 'cleaned_text.txt')
        print("✓ 文件保存成功")
        print("  - raw_text.txt (原始文本)")
        print("  - cleaned_text.txt (清洗后文本)")

        # 统计信息
        removed = len(raw_text) - len(cleaned_text)
        rate = (removed / len(raw_text) * 100) if raw_text else 0

        print("\n" + "=" * 70)
        print("处理统计:")
        print(f"  原始字符数:   {len(raw_text):>8}")
        print(f"  清洗后字符数: {len(cleaned_text):>8}")
        print(f"  移除字符数:   {removed:>8}")
        print(f"  移除率:       {rate:>7.2f}%")
        print("=" * 70)

        # 预览
        preview_len = min(500, len(raw_text))
        print(f"\n原始文本预览 (前{preview_len}字):")
        print("-" * 70)
        print(raw_text[:preview_len] + ("..." if len(raw_text) > preview_len else ""))

        preview_len = min(500, len(cleaned_text))
        print(f"\n清洗后文本预览 (前{preview_len}字):")
        print("-" * 70)
        print(cleaned_text[:preview_len] + ("..." if len(cleaned_text) > preview_len else ""))
        print("=" * 70)
        print("\n✓ 处理完成！文件已保存到当前目录。")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n建议：")
        print("  • 检查网络连接")
        print("  • 使用百度百科等静态网页")
        print("  • 确认URL格式正确")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print(" " * 15 + "网页文本爬取与清洗系统 v2.0")
    print("=" * 70)
    print("\n请选择运行模式:")
    print("  1. 图形界面 (GUI) - 推荐")
    print("  2. 命令行 (CLI)")
    print("  3. 退出")

    choice = input("\n请输入选择 (1/2/3, 默认为1): ").strip()

    if choice == '2':
        main_cli()
    elif choice == '3':
        print("再见！")
        sys.exit(0)
    else:
        main_gui()