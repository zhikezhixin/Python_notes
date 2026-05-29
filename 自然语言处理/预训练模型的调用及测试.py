# 实验三：预训练模型的调用及测试
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer
import os

# ==================== 0. 配置本地模型路径 ====================
MODEL_DIR = r"D:\CodeManager\PycharmProejcts\Mode\bert-base-chinese"  # 你的本地模型目录

# 可选：保留镜像环境变量（即使离线也不会影响）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 70)
print("实验三：预训练模型的调用及测试")
print("当前使用方式：从本地目录加载预训练模型")
print(f"本地模型路径: {MODEL_DIR}")
print("=" * 70)


# ==================== 1. 文本分词功能 ====================
def Tokenizer(model_path, text):
    """
    利用预训练模型进行分词
    @param model_path: 所需加载模型的地址（本地文件夹）
    @param text: 进行分词的文本
    @return token: 返回分词结果
    """
    # 加载模型对应的分词器（仅使用本地文件，不访问网络）
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # 对文本进行分词
    token = tokenizer.tokenize(text)

    return token


# ==================== 2. 文本特征提取网络 ====================
class TextNet(nn.Module):
    """
    基于BERT的文本特征提取网络
    """

    def __init__(self, code_length, model_path=None):
        """
        初始化网络结构
        @param code_length: 特征向量的维度
        @param model_path: 模型路径（本地文件夹）
        """
        super(TextNet, self).__init__()

        # 如果没有单独传入，就使用全局的本地模型路径
        if model_path is None:
            model_path = MODEL_DIR

        print(f"\n正在从本地路径加载BERT模型...")
        print(f"模型目录: {model_path}")

        # 加载BERT模型配置（仅本地）
        modelConfig = BertConfig.from_pretrained(model_path, local_files_only=True)

        # 加载BERT预训练模型（仅本地）
        self.textExtractor = BertModel.from_pretrained(
            model_path,
            config=modelConfig,
            local_files_only=True
        )

        # 获取BERT模型的隐藏层维度
        embedding_dim = self.textExtractor.config.hidden_size

        # 添加全连接层，调整特征维度
        self.fc = nn.Linear(embedding_dim, code_length)

        # 添加激活函数
        self.tanh = nn.Tanh()

        print(f"✓ 模型加载成功（本地）！")
        print(f"  - 隐藏层维度: {embedding_dim}")
        print(f"  - 输出特征维度: {code_length}")

    def forward(self, tokens, segments, input_masks):
        """
        前向传播函数
        @param tokens: 文本的向量表示
        @param segments: 用来指定句子编号
        @param input_masks: 用于保持输入整齐，没有实际语义
        @return features: 返回文本特征向量
        """
        # 使用BERT提取文本嵌入向量
        output = self.textExtractor(
            tokens,
            token_type_ids=segments,
            attention_mask=input_masks
        )

        # 获取[CLS]标记的输出作为句子表示
        # output[0]的形状为 [batch_size, sequence_length, hidden_size]
        text_embeddings = output[0][:, 0, :]

        # 通过全连接层调整维度
        features = self.fc(text_embeddings)

        # 通过激活函数
        features = self.tanh(features)

        return features


# ==================== 3. 主函数：测试分词和特征提取 ====================
def main():
    """
    主函数：演示预训练模型的使用（从本地加载）
    """
    model_path = MODEL_DIR

    print(f"\n{'=' * 70}")
    print("第一步：加载分词器（本地）")
    print('=' * 70)

    # 加载分词器（只用本地缓存，不访问网络）
    print(f"正在从本地加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("✓ 分词器加载成功！")

    # 测试文本
    texts = [
        "[CLS] 欢迎参加自然语言处理学习。[SEP]",
        "[CLS] 本课程可以帮助您理解深度学习技术在自然语言处理中的应用。[SEP]"
    ]

    print(f"\n{'=' * 70}")
    print("第二步：文本分词")
    print('=' * 70)
    print(f"测试文本数量: {len(texts)}\n")

    # 存储处理后的tokens、segments和input_masks
    tokens, segments, input_masks = [], [], []

    # 对每个文本进行处理
    for i, text in enumerate(texts):
        print(f"【文本 {i + 1}】")
        print(f"原文: {text}")

        # 1. 分词
        tokenized_text = tokenizer.tokenize(text)
        print(f"分词: {tokenized_text}")
        print(f"分词数: {len(tokenized_text)} 个token\n")

        # 2. 将token转换成对应的id
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)

        # 3. 创建segments（单句子全为0）
        segments.append([0] * len(indexed_tokens))

        # 4. 创建attention mask（实际token为1，padding为0）
        input_masks.append([1] * len(indexed_tokens))

    print(f"{'=' * 70}")
    print("第三步：序列填充（Padding）")
    print('=' * 70)

    # 找到最长序列的长度，用于padding
    max_len = max([len(single) for single in tokens])
    print(f"最大序列长度: {max_len}")

    # 对所有序列进行padding，使长度一致
    for j in range(len(tokens)):
        padding_length = max_len - len(tokens[j])
        if padding_length > 0:
            print(f"文本 {j + 1} 需要填充 {padding_length} 个位置")
            tokens[j] += [0] * padding_length
            segments[j] += [0] * padding_length
            input_masks[j] += [0] * padding_length

    # 转换为tensor
    tokens_tensor = torch.tensor(tokens)
    segments_tensor = torch.tensor(segments)
    input_masks_tensor = torch.tensor(input_masks)

    print(f"\n转换为Tensor:")
    print(f"  - Tokens shape: {tokens_tensor.shape}")
    print(f"  - Segments shape: {segments_tensor.shape}")
    print(f"  - Input masks shape: {input_masks_tensor.shape}")

    print(f"\n{'=' * 70}")
    print("第四步：加载BERT模型并提取特征（本地）")
    print('=' * 70)

    # 创建模型实例（输出特征维度为256）
    model = TextNet(code_length=256, model_path=model_path)

    # 设置为评估模式（不进行梯度计算）
    model.eval()

    print(f"\n开始提取文本特征...")

    # 提取特征（不计算梯度，节省内存）
    with torch.no_grad():
        features = model(tokens_tensor, segments_tensor, input_masks_tensor)

    print(f"✓ 特征提取完成！\n")

    print(f"{'=' * 70}")
    print("第五步：查看提取结果")
    print('=' * 70)
    print(f"输出特征shape: {features.shape}")
    print(f"  - batch_size (文本数量): {features.shape[0]}")
    print(f"  - feature_dim (特征维度): {features.shape[1]}")

    print(f"\n文本1的特征向量（前10个值）:")
    print(features[0][:10])

    print(f"\n文本2的特征向量（前10个值）:")
    print(features[1][:10])

    # 计算两个文本的余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        features[0].unsqueeze(0),
        features[1].unsqueeze(0)
    )
    print(f"\n两个文本的余弦相似度: {cos_sim.item():.4f}")

    return features


# ==================== 4. 模型信息查看函数 ====================
def show_model_details():
    """
    展示BERT模型的内部细节（从本地加载）
    """
    model_path = MODEL_DIR

    print(f"\n{'=' * 70}")
    print("附加信息：BERT模型详细配置（本地）")
    print('=' * 70)

    # 加载模型配置
    config = BertConfig.from_pretrained(model_path, local_files_only=True)

    print(f"\n【词表与维度】")
    print(f"  - 词表大小 (vocab_size): {config.vocab_size:,}")
    print(f"  - 隐藏层维度 (hidden_size): {config.hidden_size}")
    print(f"  - 中间层维度 (intermediate_size): {config.intermediate_size}")

    print(f"\n【模型架构】")
    print(f"  - Transformer层数 (num_hidden_layers): {config.num_hidden_layers}")
    print(f"  - 注意力头数 (num_attention_heads): {config.num_attention_heads}")
    print(f"  - 每个注意力头的维度: {config.hidden_size // config.num_attention_heads}")

    print(f"\n【序列参数】")
    print(f"  - 最大序列长度 (max_position_embeddings): {config.max_position_embeddings}")
    print(f"  - Token类型数量 (type_vocab_size): {config.type_vocab_size}")

    print(f"\n【其他参数】")
    print(f"  - Dropout率: {config.hidden_dropout_prob}")
    print(f"  - 注意力Dropout率: {config.attention_probs_dropout_prob}")

    # 加载完整模型以统计参数
    print(f"\n正在从本地加载完整模型进行参数统计...")
    model = BertModel.from_pretrained(model_path, local_files_only=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n【参数统计】")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小 (float32): {total_params * 4 / 1024 / 1024:.2f} MB")

    print(f"\n【模型结构层级】")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  - {name:20s}: {module.__class__.__name__:30s} ({param_count:,} 参数)")


# ==================== 主程序入口 ====================
if __name__ == "__main__":

    try:
        print("\n提示：当前模式为【本地加载】，不会再尝试从网络下载模型。")
        print("      如果路径配置正确，离线环境也可以顺利完成实验。\n")

        # 执行主函数
        features = main()

        # 显示模型详细信息
        show_model_details()

        print(f"\n{'=' * 70}")
        print("✓ 实验全部完成！")
        print('=' * 70)
        print("\n实验要求完成情况：")
        print("  ✓ 1. 安装和使用transformers库")
        print("  ✓ 2. 本地使用BERT预训练模型")
        print("  ✓ 3. 展示模型内部细节")
        print("  ✓ 4. 使用预训练模型进行文本分词")
        print("  ✓ 5. 实现文本的向量化表示")
        print("  ✓ 6. 使用模型进行特征提取")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print("✗ 程序执行出错")
        print('=' * 70)
        print(f"错误信息: {e}")
        print("\n排查方向：")
        print("  1. 确认 MODEL_DIR 路径是否正确、文件夹名称是否拼写错误")
        print("  2. 确认该目录下包含 config.json / pytorch_model.bin / vocab.txt 等文件")
        print("  3. 如果路径中有中文或空格，建议改成纯英文路径再试")
        print(f"\n{'=' * 70}\n")
        raise
