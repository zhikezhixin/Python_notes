import numpy as np
import matplotlib.pyplot as plt


class SimpleRNN:
    """
    简单的RNN模型实现
    用于教学演示RNN的核心原理
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化RNN参数
        :param input_size: 输入维度
        :param hidden_size: 隐藏层维度
        :param output_size: 输出维度
        :param learning_rate: 学习率
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # 初始化权重矩阵（使用Xavier初始化）
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # 输入到隐藏层
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # 隐藏层到输出

        # 偏置项
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        前向传播
        :param inputs: 输入序列 (seq_length, input_size, 1)
        :param h_prev: 前一时刻的隐藏状态
        :return: 输出序列, 隐藏状态序列
        """
        h_states = {}  # 保存每个时刻的隐藏状态
        y_outputs = {}  # 保存每个时刻的输出

        h_states[-1] = np.copy(h_prev)

        # 对序列中的每个时间步进行处理
        for t in range(len(inputs)):
            # h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            h_states[t] = np.tanh(
                np.dot(self.Wxh, inputs[t]) +
                np.dot(self.Whh, h_states[t - 1]) +
                self.bh
            )

            # y_t = Why * h_t + by
            y_outputs[t] = np.dot(self.Why, h_states[t]) + self.by

        return y_outputs, h_states

    def backward(self, inputs, h_states, y_outputs, targets):
        """
        BPTT反向传播（通过时间反向传播）
        :param inputs: 输入序列
        :param h_states: 隐藏状态序列
        :param y_outputs: 输出序列
        :param targets: 目标序列
        :return: 损失值
        """
        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(h_states[0])
        loss = 0

        # 从后向前遍历时间步
        for t in reversed(range(len(inputs))):
            # 计算输出层误差
            dy = y_outputs[t] - targets[t]
            loss += 0.5 * np.sum(dy ** 2)  # MSE损失

            # 输出层梯度
            dWhy += np.dot(dy, h_states[t].T)
            dby += dy

            # 隐藏层误差（来自输出层 + 下一时刻）
            dh = np.dot(self.Why.T, dy) + dh_next

            # tanh的导数
            dh_raw = (1 - h_states[t] ** 2) * dh

            # 隐藏层梯度
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].T)
            dWhh += np.dot(dh_raw, h_states[t - 1].T)

            # 传递到前一时刻
            dh_next = np.dot(self.Whh.T, dh_raw)

        # 梯度裁剪，防止梯度爆炸
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # 更新参数
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

        return loss

    def train(self, X_train, y_train, epochs=100, verbose=True):
        """
        训练RNN模型
        :param X_train: 训练输入序列
        :param y_train: 训练目标序列
        :param epochs: 训练轮数
        :param verbose: 是否打印训练信息
        :return: 损失历史
        """
        loss_history = []

        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden_size, 1))

            # 前向传播
            y_outputs, h_states = self.forward(X_train, h_prev)

            # 反向传播
            loss = self.backward(X_train, h_states, y_outputs, y_train)

            loss_history.append(loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return loss_history

    def predict(self, inputs, h_prev=None):
        """
        预测
        :param inputs: 输入序列
        :param h_prev: 初始隐藏状态
        :return: 预测输出
        """
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        y_outputs, _ = self.forward(inputs, h_prev)
        return y_outputs


# ============= 演示：学习简单的正弦波序列 =============
def demo_sine_wave_prediction():
    """
    演示案例：使用RNN预测正弦波序列
    """
    print("=" * 50)
    print("RNN演示：正弦波序列预测")
    print("=" * 50)

    # 生成正弦波数据
    seq_length = 50
    t = np.linspace(0, 4 * np.pi, seq_length)
    sine_wave = np.sin(t)

    # 准备训练数据（使用前面的值预测后面的值）
    X_train = []
    y_train = []

    for i in range(len(sine_wave) - 1):
        X_train.append(np.array([[sine_wave[i]]]))
        y_train.append(np.array([[sine_wave[i + 1]]]))

    # 创建并训练RNN
    rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)

    print("\n开始训练...")
    loss_history = rnn.train(X_train, y_train, epochs=200, verbose=True)

    # 测试预测
    print("\n生成预测序列...")
    predictions = []
    h_prev = np.zeros((rnn.hidden_size, 1))

    for x in X_train:
        y_pred, h_states = rnn.forward([x], h_prev)
        predictions.append(y_pred[0][0, 0])
        h_prev = h_states[0]

    # 可视化结果
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 预测结果
    plt.subplot(2, 1, 2)
    plt.plot(sine_wave[1:], label='真实值', linewidth=2)
    plt.plot(predictions, label='预测值', linestyle='--', linewidth=2)
    plt.title('RNN预测结果')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_demo_results.png', dpi=150)
    print("\n结果已保存到 'rnn_demo_results.png'")
    plt.show()

    # 计算预测精度
    mse = np.mean((np.array(predictions) - sine_wave[1:]) ** 2)
    print(f"\n最终MSE: {mse:.6f}")
    print("\n演示完成！")


if __name__ == "__main__":
    # 运行演示
    demo_sine_wave_prediction()

    print("\n" + "=" * 50)
    print("关键知识点总结：")
    print("=" * 50)
    print("1. RNN前向传播: h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)")
    print("2. 输出计算: y_t = Why·h_t + by")
    print("3. BPTT算法: 通过时间反向传播梯度")
    print("4. 梯度裁剪: 防止梯度爆炸")
    print("5. 隐藏状态: 携带历史信息的关键")
    print("=" * 50)