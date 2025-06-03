import numpy as np
import matplotlib.pyplot as plt
import struct
import random
from scipy.special import expit  # sigmoid函数的优化版本
import os


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
            if epoch % 100 == 0:
                output = self.forward(X)
                loss = np.mean(np.square(y - output))
                accuracy = self.calculate_accuracy(X, y)
                losses.append(loss)
                accuracies.append(accuracy)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

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
    # 随机选择样本
    indices = random.sample(range(len(images)), num_samples)

    # 创建10x10的子图
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
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


def main():
    """主函数"""
    print("=== 神经网络手写体数字识别实验 ===")

    # 直接设置文件路径（请修改为您的实际路径）
    base_path = r"D:\Zeker\Documents\MachineLearning\mnist\mnist"

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
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return

    try:
        # 加载数据
        print("\n正在加载数据...")
        train_images = load_mnist_images(train_images_path)
        train_labels = load_mnist_labels(train_labels_path)
        test_images = load_mnist_images(test_images_path)
        test_labels = load_mnist_labels(test_labels_path)

        if any(data is None for data in [train_images, train_labels, test_images, test_labels]):
            print("数据加载失败，请检查文件路径和文件格式")
            return

        print(f"训练集: {train_images.shape[0]} 个样本, 图像维度: {train_images.shape[1]}")
        print(f"测试集: {test_images.shape[0]} 个样本, 图像维度: {test_images.shape[1]}")
        print(f"标签范围: {train_labels.min()} - {train_labels.max()}")

        # 显示随机样本
        print("\n显示随机抽取的100个样本...")
        display_random_samples(train_images, train_labels, 100)

        # 将标签转换为one-hot编码
        print("\n转换标签为one-hot编码...")
        train_labels_onehot = one_hot_encode(train_labels)
        test_labels_onehot = one_hot_encode(test_labels)

        # 创建神经网络
        print("\n创建神经网络...")
        nn = NeuralNetwork(input_size=784, hidden_size=30, output_size=10, learning_rate=0.1)

        # 训练网络（可以先用较少的epoch测试）
        print("\n开始训练神经网络...")
        losses, accuracies = nn.train(train_images, train_labels_onehot, epochs=500, batch_size=32)

        # 评估性能
        print("\n评估网络性能...")
        train_accuracy = nn.calculate_accuracy(train_images, train_labels_onehot)
        test_accuracy = nn.calculate_accuracy(test_images, test_labels_onehot)

        print(f"\n=== 实验结果 ===")
        print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # 绘制训练过程
        if losses:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title('训练损失变化')
            plt.xlabel('训练轮次 (x100)')
            plt.ylabel('损失值')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(accuracies)
            plt.title('训练准确率变化')
            plt.xlabel('训练轮次 (x100)')
            plt.ylabel('准确率')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # 显示一些预测结果
        print("\n显示部分预测结果...")
        sample_indices = random.sample(range(len(test_images)), 20)

        fig, axes = plt.subplots(4, 5, figsize=(12, 8))
        fig.suptitle('预测结果示例', fontsize=16)

        for i, idx in enumerate(sample_indices):
            row = i // 5
            col = i % 5

            image = test_images[idx].reshape(28, 28)
            true_label = test_labels[idx]
            pred_label = nn.predict(test_images[idx:idx + 1])[0]

            axes[row, col].imshow(image, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[row, col].set_title(f'真实: {true_label}, 预测: {pred_label}',
                                     color=color, fontsize=10)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()