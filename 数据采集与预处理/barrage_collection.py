"""
B站弹幕数据采集与分析系统
项目：《红海行动》弹幕数据挖掘与可视化
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BilibiliDanmakuAnalyzer:
    """B站弹幕分析器"""

    def __init__(self, bvid):
        """
        初始化分析器
        :param bvid: B站视频BV号
        """
        self.bvid = bvid
        self.cid = None
        self.danmaku_data = []
        self.df = None

    def get_cid(self):
        """获取视频的cid - 改进版"""
        url = f'https://api.bilibili.com/x/web-interface/view?bvid={self.bvid}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        try:
            print(f"正在获取视频信息: {self.bvid}")
            response = requests.get(url, headers=headers, timeout=10)

            # 打印响应状态
            print(f"响应状态码: {response.status_code}")

            if response.status_code != 200:
                print(f"HTTP错误: {response.status_code}")
                return False

            data = response.json()
            print(f"API返回: {data.get('message', 'success')}")

            if data['code'] == 0:
                self.cid = data['data']['cid']
                video_title = data['data']['title']
                print(f"✓ 成功获取视频信息")
                print(f"  标题: {video_title}")
                print(f"  CID: {self.cid}")
                return True
            else:
                print(f"✗ API返回错误码: {data['code']}")
                print(f"  错误信息: {data.get('message', '未知错误')}")
                return False

        except requests.exceptions.Timeout:
            print("✗ 请求超时，请检查网络连接")
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ 网络请求错误: {e}")
            return False
        except Exception as e:
            print(f"✗ 获取CID出错: {e}")
            return False

    def fetch_danmaku(self):
        """采集弹幕数据 - 改进版"""
        if not self.cid:
            print("请先获取CID")
            return False

        url = f'https://api.bilibili.com/x/v1/dm/list.so?oid={self.cid}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept-Encoding': 'gzip, deflate',
        }

        try:
            print(f"\n正在采集弹幕数据...")
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                print(f"✗ 弹幕请求失败: HTTP {response.status_code}")
                return False

            # 检查是否是XML格式
            if not response.text.strip().startswith('<?xml'):
                print("✗ 返回的不是XML格式数据")
                print(f"返回内容前100字符: {response.text[:100]}")
                return False

            # 解析XML格式的弹幕数据
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)

            danmaku_count = 0
            for d in root.findall('d'):
                try:
                    p = d.attrib['p'].split(',')
                    if len(p) >= 8 and d.text:  # 确保数据完整
                        danmaku_info = {
                            'time': float(p[0]),  # 弹幕出现时间(秒)
                            'mode': int(p[1]),  # 弹幕类型
                            'fontsize': int(p[2]),  # 字体大小
                            'color': int(p[3]),  # 颜色
                            'timestamp': int(p[4]),  # 发送时间戳
                            'pool': int(p[5]),  # 弹幕池
                            'user_id': p[6],  # 用户ID
                            'rowID': p[7],  # 弹幕ID
                            'content': d.text.strip()  # 弹幕内容
                        }
                        self.danmaku_data.append(danmaku_info)
                        danmaku_count += 1
                except (IndexError, ValueError) as e:
                    continue  # 跳过格式错误的弹幕

            if danmaku_count > 0:
                print(f"✓ 成功采集 {danmaku_count} 条弹幕")
                return True
            else:
                print("✗ 未采集到任何弹幕数据")
                return False

        except ET.ParseError as e:
            print(f"✗ XML解析错误: {e}")
            return False
        except Exception as e:
            print(f"✗ 采集弹幕出错: {e}")
            return False

    def preprocess_data(self):
        """数据预处理"""
        if not self.danmaku_data:
            print("没有弹幕数据")
            return False

        print(f"\n开始数据预处理...")

        # 转换为DataFrame
        self.df = pd.DataFrame(self.danmaku_data)

        # 时间戳转换为日期时间
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
        self.df['date'] = self.df['datetime'].dt.date
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute

        # 弹幕出现时间转换为分钟
        self.df['video_minute'] = self.df['time'] // 60

        # 内容清洗
        self.df['content_clean'] = self.df['content'].apply(self.clean_text)

        # 过滤空内容
        original_count = len(self.df)
        self.df = self.df[self.df['content_clean'].str.len() > 0]
        filtered_count = original_count - len(self.df)

        # 内容长度
        self.df['length'] = self.df['content'].apply(len)

        print(f"✓ 数据预处理完成")
        print(f"  原始数据: {original_count} 条")
        print(f"  过滤空内容: {filtered_count} 条")
        print(f"  有效数据: {len(self.df)} 条")
        print(f"\n数据样例:")
        print(self.df[['content', 'datetime', 'video_minute']].head(3))

        return True

    def clean_text(self, text):
        """文本清洗"""
        if pd.isna(text):
            return ""
        # 去除特殊字符和表情
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', str(text))
        return text.strip()

    def analyze_basic_info(self):
        """分析基本信息"""
        print("\n" + "=" * 50)
        print("弹幕基本信息分析")
        print("=" * 50)

        print(f"弹幕总数: {len(self.df)}")
        print(f"发送用户数: {self.df['user_id'].nunique()}")
        print(f"人均弹幕数: {len(self.df) / self.df['user_id'].nunique():.2f} 条")
        print(f"平均弹幕长度: {self.df['length'].mean():.2f} 字")
        print(f"最长弹幕: {self.df['length'].max()} 字")
        print(f"最短弹幕: {self.df['length'].min()} 字")

        # 弹幕类型分布
        print("\n弹幕类型分布:")
        mode_map = {1: '普通弹幕', 4: '底部弹幕', 5: '顶部弹幕', 6: '逆向弹幕', 7: '精准定位', 8: '高级弹幕'}
        mode_counts = self.df['mode'].value_counts()
        for mode, count in mode_counts.items():
            percentage = count / len(self.df) * 100
            print(f"  {mode_map.get(mode, f'类型{mode}')}: {count} 条 ({percentage:.1f}%)")

    def analyze_time_distribution(self):
        """分析时间分布"""
        print("\n" + "=" * 50)
        print("弹幕时间分布分析")
        print("=" * 50)

        # 按日期统计
        daily_counts = self.df.groupby('date').size().sort_index()
        print(f"\n发送日期范围: {daily_counts.index[0]} 至 {daily_counts.index[-1]}")
        print(f"发送天数: {len(daily_counts)} 天")
        print(f"首日弹幕量: {daily_counts.iloc[0]} 条")
        print(f"峰值日期: {daily_counts.idxmax()} ({daily_counts.max()} 条)")

        # 按小时统计
        hourly_counts = self.df.groupby('hour').size()
        peak_hour = hourly_counts.idxmax()
        print(f"\n活跃时段: {peak_hour}:00 时 (共 {hourly_counts[peak_hour]} 条弹幕)")
        print(f"低谷时段: {hourly_counts.idxmin()}:00 时 (共 {hourly_counts[hourly_counts.idxmin()]} 条弹幕)")

        # 视频时间轴
        video_time_counts = self.df.groupby('video_minute').size()
        peak_minute = video_time_counts.idxmax()
        print(f"\n视频高潮时刻: 第 {peak_minute} 分钟 ({video_time_counts[peak_minute]} 条弹幕)")

        return daily_counts, hourly_counts

    def extract_keywords(self, top_n=30):
        """提取关键词"""
        print("\n" + "=" * 50)
        print("提取弹幕关键词")
        print("=" * 50)

        # 停用词
        stopwords = {'的', '了', '是', '我', '你', '他', '在', '有', '这', '个', '也', '就',
                     '不', '和', '都', '啊', '吗', '吧', '哈', '呀', '么', '呢', '哦', '嗯', '额'}

        # 分词
        print("正在分词...")
        all_words = []
        for content in self.df['content_clean']:
            words = jieba.lcut(content)
            all_words.extend([w for w in words if len(w) >= 2 and w not in stopwords])

        # 词频统计
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)

        print(f"\nTop {min(10, len(top_words))} 关键词:")
        for i, (word, count) in enumerate(top_words[:10], 1):
            print(f"  {i}. {word}: {count} 次")

        return top_words

    def visualize_results(self, daily_counts, hourly_counts, top_words):
        """可视化分析结果"""
        print("\n" + "=" * 50)
        print("生成可视化图表")
        print("=" * 50)

        fig = plt.figure(figsize=(16, 12))

        # 1. 弹幕数量与日期关系
        ax1 = plt.subplot(2, 3, 1)
        daily_counts.plot(kind='line', ax=ax1, marker='o', color='#5B9BD5', linewidth=2, markersize=6)
        ax1.set_title('弹幕数量-日期趋势图', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('日期', fontsize=11)
        ax1.set_ylabel('弹幕数量', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. 弹幕数量与小时关系
        ax2 = plt.subplot(2, 3, 2)
        hourly_counts.plot(kind='bar', ax=ax2, color='#70AD47', width=0.8)
        ax2.set_title('弹幕数量-小时分布图', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('小时', fontsize=11)
        ax2.set_ylabel('弹幕数量', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # 3. 视频时间轴弹幕分布
        ax3 = plt.subplot(2, 3, 3)
        video_time_counts = self.df.groupby('video_minute').size()
        video_time_counts.plot(kind='area', ax=ax3, color='#FFC000', alpha=0.7, linewidth=2)
        ax3.set_title('视频时间轴弹幕密度图', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xlabel('视频时长(分钟)', fontsize=11)
        ax3.set_ylabel('弹幕数量', fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 4. 高频词Top20柱状图
        ax4 = plt.subplot(2, 3, 4)
        words, counts = zip(*top_words[:20])
        y_pos = range(len(words))
        ax4.barh(y_pos, counts, color='#C55A11', height=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(words, fontsize=9)
        ax4.set_title('Top20高频词统计', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xlabel('出现次数', fontsize=11)
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x', linestyle='--')

        # 5. 弹幕长度分布
        ax5 = plt.subplot(2, 3, 5)
        self.df['length'].hist(bins=30, ax=ax5, color='#9966FF', edgecolor='black', alpha=0.7)
        ax5.set_title('弹幕长度分布直方图', fontsize=14, fontweight='bold', pad=10)
        ax5.set_xlabel('弹幕字数', fontsize=11)
        ax5.set_ylabel('数量', fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax5.axvline(self.df['length'].mean(), color='red', linestyle='--',
                    label=f'平均长度: {self.df["length"].mean():.1f}字')
        ax5.legend()

        # 6. 词云图
        ax6 = plt.subplot(2, 3, 6)
        try:
            wordcloud_text = ' '.join([w[0] for w in top_words])
            wordcloud = WordCloud(
                # font_path='simhei.ttf',  # 如果没有字体文件可注释此行
                width=800,
                height=600,
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate(wordcloud_text)
            ax6.imshow(wordcloud, interpolation='bilinear')
            ax6.set_title('弹幕关键词云图', fontsize=14, fontweight='bold', pad=10)
            ax6.axis('off')
        except Exception as e:
            ax6.text(0.5, 0.5, f'词云生成失败\n{str(e)}',
                     ha='center', va='center', fontsize=12)
            ax6.axis('off')

        plt.tight_layout(pad=2.0)
        plt.savefig('弹幕分析结果.png', dpi=300, bbox_inches='tight')
        print("✓ 可视化图表已保存: 弹幕分析结果.png")
        plt.show()

    def generate_report_data(self):
        """生成报告所需数据"""
        print("\n" + "=" * 50)
        print("生成报告数据文件")
        print("=" * 50)

        # 保存原始数据样本
        sample_size = min(100, len(self.df))
        self.df.head(sample_size).to_csv('弹幕数据样本.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: 弹幕数据样本.csv ({sample_size}条)")

        # 保存预处理后的数据
        self.df.to_csv('弹幕数据_预处理完成.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: 弹幕数据_预处理完成.csv ({len(self.df)}条)")

        # 统计数据
        stats = {
            '弹幕总数': len(self.df),
            '用户数': self.df['user_id'].nunique(),
            '人均弹幕': round(len(self.df) / self.df['user_id'].nunique(), 2),
            '平均长度': round(self.df['length'].mean(), 2),
            '最长弹幕': self.df['length'].max(),
            '最短弹幕': self.df['length'].min(),
            '发送天数': self.df['date'].nunique(),
            '视频时长_分钟': round(self.df['time'].max() / 60, 2)
        }

        pd.DataFrame([stats]).to_csv('统计数据.csv', index=False, encoding='utf-8-sig')
        print("✓ 已保存: 统计数据.csv")

        print("\n数据文件生成完成！可以将这些文件插入报告中。")


def main():
    """主程序"""
    print("=" * 50)
    print("B站弹幕数据采集与分析系统")
    print("=" * 50)

    # 推荐的热门视频BV号（可替换）
    print("\n提示：请输入视频BV号，或直接回车使用默认视频")
    print("推荐视频：")
    print("  1. BV1uW411P7T8 - 知名热门视频")
    print("  2. BV1xx411c7mD - 高播放量视频")
    print("  3. 自定义输入")

    bvid_input = input("\n请输入BV号 (直接回车使用默认): ").strip()

    if not bvid_input:
        bvid = "BV1uW411P7T8"  # 默认视频
        print(f"使用默认BV号: {bvid}")
    else:
        bvid = bvid_input

    # 创建分析器
    analyzer = BilibiliDanmakuAnalyzer(bvid)

    # 1. 获取CID
    print("\n" + "-" * 50)
    if not analyzer.get_cid():
        print("\n" + "=" * 50)
        print("程序终止：无法获取视频信息")
        print("=" * 50)
        print("\n可能的原因:")
        print("  1. BV号不正确")
        print("  2. 视频已被删除或设为私密")
        print("  3. 网络连接问题")
        print("\n解决方案:")
        print("  1. 检查BV号是否正确（从B站视频链接复制）")
        print("  2. 尝试其他视频的BV号")
        print("  3. 检查网络连接是否正常")
        return

    # 2. 采集弹幕
    print("\n" + "-" * 50)
    if not analyzer.fetch_danmaku():
        print("\n弹幕采集失败，可能原因：")
        print("  1. 该视频没有弹幕")
        print("  2. 弹幕接口限制")
        print("  3. 网络问题")
        return

    # 3. 数据预处理
    print("\n" + "-" * 50)
    if not analyzer.preprocess_data():
        print("数据预处理失败")
        return

    # 4. 基本信息分析
    analyzer.analyze_basic_info()

    # 5. 时间分布分析
    daily_counts, hourly_counts = analyzer.analyze_time_distribution()

    # 6. 关键词提取
    top_words = analyzer.extract_keywords(top_n=30)

    # 7. 可视化
    analyzer.visualize_results(daily_counts, hourly_counts, top_words)

    # 8. 生成报告数据
    analyzer.generate_report_data()

    print("\n" + "=" * 50)
    print("✓ 分析完成！")
    print("=" * 50)
    print("\n生成的文件:")
    print("  - 弹幕数据样本.csv")
    print("  - 弹幕数据_预处理完成.csv")
    print("  - 统计数据.csv")
    print("  - 弹幕分析结果.png")
    print("\n这些文件可以直接插入到报告中使用。")


if __name__ == "__main__":
    main()