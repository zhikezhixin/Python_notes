import numpy as np
import matplotlib.pyplot as plt
import struct
import random
from scipy.special import expit  # sigmoid函数的优化版本
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=30, output_size=10, learning_rate=0.1):
        """
        初始化神经网络
        input_size: 输入层神经元数量 (28*28=784)
        hidden_size: 隐含层神经元数量
        output_size: 输出层神经元数量 (10个数字类别)
        learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        # 使用Xavier初始化方法
        self.W1 = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        print(f"神经网络结构:")
        print(f"输入层: {self.input_size} 个神经元")
        print(f"隐含层: {self.hidden_size} 个神经元 (sigmoid激活)")
        print(f"输出层: {self.output_size} 个神经元 (sigmoid激活)")
        print(f"学习率: {self.learning_rate}")

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return expit(x)  # 使用scipy的优化版本，避免溢出

    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数"""
        return x * (1 - x)

    def forward(self, X):
        """前向传播"""
        # 输入层到隐含层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 隐含层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]

        # 计算输出层误差
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        # 计算隐含层误差
        self.hidden_error = self.output_delta.dot(self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)

        # 更新权重和偏置
        self.W2 += self.a1.T.dot(self.output_delta) * self.learning_rate / m
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate / m
        self.W1 += X.T.dot(self.hidden_delta) * self.learning_rate / m
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate / m

    def train(self, X, y, epochs=1000, batch_size=32):
        """训练神经网络"""
        losses = []
        accuracies = []

        print(f"\n开始训练神经网络...")
        print(f"训练样本数: {X.shape[0]}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 分批训练
            for i in range(0, X.shape[0], batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                # 前向传播
                output = self.forward(batch_X)

                # 反向传播
                self.backward(batch_X, batch_y, output)

            # 计算整体损失和准确率
            if epoch % 50 == 0:
                output = self.forward(X)
                loss = np.mean(np.square(y - output))
                accuracy = self.calculate_accuracy(X, y)
                losses.append(loss)
                accuracies.append(accuracy)
                print(f'Epoch {epoch:4d}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')

        return losses, accuracies

    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def calculate_accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        y_labels = np.argmax(y, axis=1)
        return np.mean(predictions == y_labels)


def load_mnist_images(filename):
    """加载MNIST图像文件"""
    try:
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            print(f"加载图像文件: Magic={magic}, 数量={num}, 行={rows}, 列={cols}")
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            # 归一化到0-1范围
            images = images.astype(np.float32) / 255.0
        return images
    except Exception as e:
        print(f"加载图像文件出错: {e}")
        return None


def load_mnist_labels(filename):
    """加载MNIST标签文件"""
    try:
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            print(f"加载标签文件: Magic={magic}, 数量={num}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    except Exception as e:
        print(f"加载标签文件出错: {e}")
        return None


def one_hot_encode(labels, num_classes=10):
    """将标签转换为one-hot编码"""
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded


def display_random_samples(images, labels, num_samples=100):
    """显示随机抽取的样本"""
    print(f"\n正在显示随机抽取的{num_samples}个样本...")

    # 随机选择样本
    indices = random.sample(range(len(images)), num_samples)

    # 创建10x10的子图
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle('随机抽取的100个手写数字样本', fontsize=16)

    for i, idx in enumerate(indices):
        row = i // 10
        col = i % 10

        # 将784维向量重塑为28x28图像
        image = images[idx].reshape(28, 28)

        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'标签: {labels[idx]}', fontsize=8)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(losses, accuracies):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失函数图
    axes[0].plot(range(0, len(losses) * 50, 50), losses, 'b-', linewidth=2)
    axes[0].set_title('训练损失变化', fontsize=14)
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('损失值')
    axes[0].grid(True, alpha=0.3)

    # 准确率图
    axes[1].plot(range(0, len(accuracies) * 50, 50), [acc * 100 for acc in accuracies], 'r-', linewidth=2)
    axes[1].set_title('训练准确率变化', fontsize=14)
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_predictions(images, true_labels, nn, num_samples=20):
    """显示预测结果"""
    print(f"\n显示{num_samples}个预测结果示例...")

    sample_indices = random.sample(range(len(images)), num_samples)

    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    fig.suptitle('预测结果示例 (绿色=正确, 红色=错误)', fontsize=16)

    correct_count = 0
    for i, idx in enumerate(sample_indices):
        row = i // 5
        col = i % 5

        image = images[idx].reshape(28, 28)
        true_label = true_labels[idx]
        pred_label = nn.predict(images[idx:idx + 1])[0]

        if true_label == pred_label:
            correct_count += 1

        axes[row, col].imshow(image, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[row, col].set_title(f'真实: {true_label}, 预测: {pred_label}',
                                 color=color, fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"示例中正确预测: {correct_count}/{num_samples} ({correct_count / num_samples * 100:.1f}%)")


def analyze_confusion_matrix(true_labels, predictions):
    """分析混淆矩阵"""
    print("\n=== 混淆矩阵分析 ===")

    # 计算混淆矩阵
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(true_labels, predictions):
        confusion_matrix[true, pred] += 1

    # 计算每个类别的准确率
    print("\n各数字识别准确率:")
    for i in range(10):
        if np.sum(confusion_matrix[i, :]) > 0:
            accuracy = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            print(f"数字 {i}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 找出最容易混淆的组合
    print("\n最容易混淆的数字组合:")
    for i in range(10):
        for j in range(10):
            if i != j and confusion_matrix[i, j] > 0:
                error_rate = confusion_matrix[i, j] / np.sum(confusion_matrix[i, :])
                if error_rate > 0.05:  # 错误率超过5%
                    print(f"数字 {i} 被误识别为 {j}: {error_rate:.3f} ({error_rate * 100:.1f}%)")


def main():
    """主函数"""
    print("=== 神经网络手写体数字识别实验 ===")
    print("实验要求：使用三层神经网络识别MNIST手写数字")
    print("网络结构：784输入 -> 30隐含 -> 10输出")

    # 文件路径设置
    base_path = r"D:\Zeker\Documents\机器学习\实验数据\mnist\mnist"

    train_images_path = os.path.join(base_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(base_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(base_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(base_path, "t10k-labels.idx1-ubyte")

    print(f"\n使用的文件路径:")
    print(f"训练图像: {train_images_path}")
    print(f"训练标签: {train_labels_path}")
    print(f"测试图像: {test_images_path}")
    print(f"测试标签: {test_labels_path}")

    # 检查文件是否存在
    files_to_check = [train_images_path, train_labels_path, test_images_path, test_labels_path]
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"\n错误: 以下文件不存在：")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n建议：")
        print("1. 检查文件路径是否正确")
        print("2. 确认MNIST数据集已正确下载")
        print("3. 数据文件应包含：")
        print("   - train-images.idx3-ubyte")
        print("   - train-labels.idx1-ubyte")
        print("   - t10k-images.idx3-ubyte")
        print("   - t10k-labels.idx1-ubyte")
        return

    try:
        # 加载数据
        print("\n=== 第一步：数据加载 ===")
        train_images = load_mnist_images(train_images_path)
        train_labels = load_mnist_labels(train_labels_path)
        test_images = load_mnist_images(test_images_path)
        test_labels = load_mnist_labels(test_labels_path)

        if any(data is None for data in [train_images, train_labels, test_images, test_labels]):
            print("数据加载失败，请检查文件路径和文件格式")
            return

        print(f"\n数据加载成功！")
        print(f"训练集: {train_images.shape[0]} 个样本, 图像维度: {train_images.shape[1]}")
        print(f"测试集: {test_images.shape[0]} 个样本, 图像维度: {test_images.shape[1]}")
        print(f"标签范围: {train_labels.min()} - {train_labels.max()}")

        # 显示随机样本
        print("\n=== 第二步：数据可视化 ===")
        display_random_samples(train_images, train_labels, 100)

        # 将标签转换为one-hot编码
        print("\n=== 第三步：数据预处理 ===")
        print("转换标签为one-hot编码...")
        train_labels_onehot = one_hot_encode(train_labels)
        test_labels_onehot = one_hot_encode(test_labels)
        print("标签转换完成")

        # 创建神经网络
        print("\n=== 第四步：创建神经网络 ===")
        nn = NeuralNetwork(input_size=784, hidden_size=30, output_size=10, learning_rate=0.1)

        # 训练网络
        print("\n=== 第五步：训练神经网络 ===")
        losses, accuracies = nn.train(train_images, train_labels_onehot, epochs=300, batch_size=32)

        # 评估性能
        print("\n=== 第六步：性能评估 ===")
        print("正在计算准确率...")
        train_accuracy = nn.calculate_accuracy(train_images, train_labels_onehot)
        test_accuracy = nn.calculate_accuracy(test_images, test_labels_onehot)

        print(f"\n=== 实验结果 ===")
        print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # 计算泛化误差
        generalization_gap = train_accuracy - test_accuracy
        print(f"泛化差距: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")

        # 绘制训练过程
        print("\n=== 第七步：结果可视化 ===")
        if losses:
            plot_training_history(losses, accuracies)

        # 显示预测结果
        display_predictions(test_images, test_labels, nn, 20)

        # 混淆矩阵分析
        test_predictions = nn.predict(test_images)
        analyze_confusion_matrix(test_labels, test_predictions)

        print(f"\n=== 实验总结 ===")
        print(f"本次实验成功实现了三层神经网络的手写体数字识别")
        print(f"网络参数量: {784 * 30 + 30 + 30 * 10 + 10} = {784 * 30 + 30 + 30 * 10 + 10}")
        print(f"训练时间: 约 {len(losses) * 50} 个epoch")
        print(f"最终性能: 训练集 {train_accuracy * 100:.2f}%, 测试集 {test_accuracy * 100:.2f}%")

        if test_accuracy > 0.9:
            print("✅ 实验成功！测试准确率超过90%")
        elif test_accuracy > 0.8:
            print("⚠️ 实验基本成功！测试准确率超过80%，但还有改进空间")
        else:
            print("❌ 实验需要改进！测试准确率低于80%")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()