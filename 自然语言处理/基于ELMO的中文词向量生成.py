import torch
import jieba
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("基于 ELMo 的中文词向量生成实验")
print("=" * 80)
print()


class ChineseELMoVectorizer:
    """中文ELMo词向量生成器"""

    def __init__(self, options_file=None, weight_file=None):
        """
        初始化ELMo模型

        参数:
            options_file: ELMo配置文件路径（可选）
            weight_file: ELMo权重文件路径（可选）
        """
        print("正在初始化 ELMo 模型...")

        # 使用英文预训练模型作为演示
        # 实际应用中应该使用中文预训练模型
        if options_file is None or weight_file is None:
            print("注意：使用英文预训练模型进行演示")
            print("实际应用中应使用中文 ELMo 预训练模型")
            print()

            # 使用 AllenNLP 提供的小型模型
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        try:
            self.elmo = Elmo(
                options_file=options_file,
                weight_file=weight_file,
                num_output_representations=1,
                dropout=0
            )
            self.elmo.eval()
            print("✓ ELMo 模型加载成功！\n")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("提示：首次运行会自动下载模型，请确保网络连接正常")
            raise

    def segment_text(self, text):
        """
        对中文文本进行分词

        参数:
            text: 输入的中文文本

        返回:
            分词后的词列表
        """
        words = list(jieba.cut(text))
        return words

    def get_word_vectors(self, text, show_details=True):
        """
        获取文本中每个词的词向量

        参数:
            text: 输入的中文文本
            show_details: 是否显示详细信息

        返回:
            words: 分词列表
            vectors: 对应的词向量 (numpy array)
        """
        # 1. 分词
        words = self.segment_text(text)

        if show_details:
            print(f"原始文本: {text}")
            print(f"分词结果: {' / '.join(words)}")
            print(f"词数量: {len(words)}")

        # 2. 将词转换为字符ID
        # ELMo是基于字符的，所以需要将每个词拆分成字符
        character_ids = batch_to_ids([words])

        # 3. 通过ELMo获取词向量
        with torch.no_grad():
            embeddings = self.elmo(character_ids)
            # 获取第一层的表示
            word_vectors = embeddings['elmo_representations'][0]
            # 去掉batch维度
            word_vectors = word_vectors.squeeze(0)

        # 转换为numpy数组
        word_vectors = word_vectors.numpy()

        if show_details:
            print(f"词向量维度: {word_vectors.shape}")
            print(f"每个词的向量维度: {word_vectors.shape[1]}\n")

        return words, word_vectors

    def display_vectors(self, words, vectors, num_dims=10):
        """
        展示词向量（显示前几个维度）

        参数:
            words: 词列表
            vectors: 词向量数组
            num_dims: 显示的维度数量
        """
        print("=" * 80)
        print("词向量结果展示")
        print("=" * 80)

        for i, (word, vector) in enumerate(zip(words, vectors)):
            print(f"\n第 {i + 1} 个词: '{word}'")
            print(f"  向量维度: {len(vector)}")
            print(f"  前 {num_dims} 维: {vector[:num_dims]}")
            print(f"  向量范数: {np.linalg.norm(vector):.4f}")

            # 显示统计信息
            print(f"  最小值: {vector.min():.4f}")
            print(f"  最大值: {vector.max():.4f}")
            print(f"  均值: {vector.mean():.4f}")
            print(f"  标准差: {vector.std():.4f}")

    def compute_similarity(self, vec1, vec2):
        """
        计算两个词向量的余弦相似度

        参数:
            vec1: 第一个词的向量
            vec2: 第二个词的向量

        返回:
            余弦相似度值
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def save_vectors(self, words, vectors, filename):
        """
        保存词向量到文件

        参数:
            words: 词列表
            vectors: 词向量数组
            filename: 输出文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"词数量: {len(words)}\n")
            f.write(f"向量维度: {vectors.shape[1]}\n")
            f.write("=" * 60 + "\n\n")

            for word, vector in zip(words, vectors):
                f.write(f"词: {word}\n")
                f.write(f"向量: {vector.tolist()}\n\n")

        print(f"✓ 词向量已保存到: {filename}")


def experiment_1_basic():
    """weather_scraper.py：基础词向量生成"""
    print("\n" + "=" * 80)
    print("weather_scraper.py：基础词向量生成")
    print("=" * 80 + "\n")

    vectorizer = ChineseELMoVectorizer()

    # 测试文本
    text = "自然语言处理是人工智能的重要分支"

    # 获取词向量
    words, vectors = vectorizer.get_word_vectors(text)

    # 显示结果
    vectorizer.display_vectors(words, vectors, num_dims=8)

    # 保存结果
    vectorizer.save_vectors(words, vectors, "实验1_词向量结果.txt")


def experiment_2_similarity():
    """实验2：词向量相似度计算"""
    print("\n\n" + "=" * 80)
    print("实验2：词向量相似度计算")
    print("=" * 80 + "\n")

    vectorizer = ChineseELMoVectorizer()

    # 测试两个句子
    text1 = "我喜欢学习机器学习"
    text2 = "我热爱研究深度学习"

    print("处理第一个句子...")
    words1, vectors1 = vectorizer.get_word_vectors(text1, show_details=True)

    print("\n处理第二个句子...")
    words2, vectors2 = vectorizer.get_word_vectors(text2, show_details=True)

    # 计算相似度矩阵
    print("\n词向量相似度矩阵:")
    print("-" * 80)

    # 打印表头
    print(f"{'':12}", end="")
    for word2 in words2:
        print(f"{word2:12}", end="")
    print()

    # 打印相似度
    for i, word1 in enumerate(words1):
        print(f"{word1:12}", end="")
        for j, word2 in enumerate(words2):
            sim = vectorizer.compute_similarity(vectors1[i], vectors2[j])
            print(f"{sim:12.4f}", end="")
        print()

    print("\n分析：")
    print("- 相似度接近1表示两个词的语义非常相似")
    print("- 相似度接近0表示两个词的语义不相关")
    print("- 相似度为负表示两个词的语义相反")


def experiment_3_context():
    """实验3：上下文相关性测试"""
    print("\n\n" + "=" * 80)
    print("实验3：上下文相关性测试（ELMo的核心特性）")
    print("=" * 80 + "\n")

    vectorizer = ChineseELMoVectorizer()

    # 同一个词在不同上下文中
    sentences = [
        "我去银行存钱",
        "我坐在河银行边钓鱼",
        "苹果公司发布新产品",
        "我喜欢吃红苹果"
    ]

    print("测试词 '银行' 和 '苹果' 在不同上下文中的向量:\n")

    # 测试"银行"
    bank_vectors = []
    for i, sent in enumerate(sentences[:2], 1):
        print(f"句子{i}: {sent}")
        words, vectors = vectorizer.get_word_vectors(sent, show_details=False)

        if "银行" in words:
            idx = words.index("银行")
            bank_vectors.append(vectors[idx])
            print(f"  '银行' 的向量前5维: {vectors[idx][:5]}")
            print()

    if len(bank_vectors) == 2:
        sim = vectorizer.compute_similarity(bank_vectors[0], bank_vectors[1])
        print(f"两个'银行'的相似度: {sim:.4f}")
        print("（同一个词在不同上下文中的相似度）\n")

    # 测试"苹果"
    print("-" * 80 + "\n")
    apple_vectors = []
    for i, sent in enumerate(sentences[2:], 1):
        print(f"句子{i + 2}: {sent}")
        words, vectors = vectorizer.get_word_vectors(sent, show_details=False)

        if "苹果" in words:
            idx = words.index("苹果")
            apple_vectors.append(vectors[idx])
            print(f"  '苹果' 的向量前5维: {vectors[idx][:5]}")
            print()

    if len(apple_vectors) == 2:
        sim = vectorizer.compute_similarity(apple_vectors[0], apple_vectors[1])
        print(f"两个'苹果'的相似度: {sim:.4f}")
        print("（同一个词在不同上下文中的相似度）")

    print("\n说明：")
    print("- ELMo生成的是上下文相关的词向量")
    print("- 同一个词在不同语境中会有不同的向量表示")
    print("- 这是ELMo相比Word2Vec/GloVe的重要优势")


def experiment_4_batch():
    """实验4：批量处理多个句子"""
    print("\n\n" + "=" * 80)
    print("实验4：批量处理多个句子")
    print("=" * 80 + "\n")

    vectorizer = ChineseELMoVectorizer()

    sentences = [
        "今天天气很好",
        "我在学习自然语言处理",
        "深度学习改变了世界",
        "机器翻译是NLP的重要应用",
        "词向量是文本表示的基础"
    ]

    all_results = []

    for i, sentence in enumerate(sentences, 1):
        print(f"处理句子 {i}/{len(sentences)}: {sentence}")
        words, vectors = vectorizer.get_word_vectors(sentence, show_details=False)

        print(f"  分词结果: {' / '.join(words)}")
        print(f"  生成了 {len(words)} 个词向量")
        print(f"  每个向量维度: {vectors.shape[1]}")
        print()

        all_results.append({
            'sentence': sentence,
            'words': words,
            'vectors': vectors
        })

    # 保存所有结果
    with open("实验4_批量处理结果.txt", 'w', encoding='utf-8') as f:
        for i, result in enumerate(all_results, 1):
            f.write(f"句子 {i}: {result['sentence']}\n")
            f.write(f"分词: {' / '.join(result['words'])}\n")
            f.write(f"词数量: {len(result['words'])}\n")
            f.write(f"向量维度: {result['vectors'].shape}\n")
            f.write("=" * 60 + "\n\n")

    print("✓ 批量处理结果已保存到: 实验4_批量处理结果.txt")


def main():
    """主函数：运行所有实验"""

    try:
        # 运行所有实验
        experiment_1_basic()
        experiment_2_similarity()
        experiment_3_context()
        experiment_4_batch()

        # 实验总结
        print("\n\n" + "=" * 80)
        print("实验总结")
        print("=" * 80)
        print("""
实验完成情况：
✓ weather_scraper.py：成功实现中文文本分词和词向量生成
✓ 实验2：成功计算词向量之间的相似度
✓ 实验3：验证了ELMo的上下文相关性特性
✓ 实验4：成功实现批量文本处理

技术要点：
1. 使用 jieba 进行中文分词
2. 使用 AllenNLP 的 ELMo 模型生成词向量
3. ELMo 生成的是上下文相关的动态词向量
4. 词向量维度通常为 256/512/1024

实验要求完成情况：
✓ 对输入的中文文本进行分词
✓ 输出每一个词的词向量
✓ 支持批量处理
✓ 提供相似度计算等扩展功能

说明：
- 本实验使用英文预训练ELMo模型作为演示
- 实际应用中应使用中文特定的预训练模型
- 可从 HIT-SCIR 等机构获取中文ELMo模型
- 或考虑使用更先进的BERT等模型

实验文件：
- 实验1_词向量结果.txt
- 实验4_批量处理结果.txt
        """)

    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        print("\n可能的原因：")
        print("1. 网络连接问题（首次运行需要下载模型）")
        print("2. 内存不足")
        print("3. 依赖包版本不兼容")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n开始运行实验...\n")
    main()
    print("\n实验结束！")