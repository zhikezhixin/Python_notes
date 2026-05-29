import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# =========================================================
# 设置中文显示
# =========================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 一、线性回归手动实现
# =========================================================

print("=" * 50)
print("线性回归实验开始")
print("=" * 50)

# 生成数据集
X_reg, y_reg = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# 转换维度
X_reg = X_reg.reshape(-1, 1)
y_reg = y_reg.reshape(-1, 1)

# 参数初始化
w = np.random.randn(1, 1)
b = 0

# 超参数
lr = 0.01
epochs = 1000

# 保存损失
loss_history = []

# 训练过程
for epoch in range(epochs):

    # 前向传播
    y_pred = X_reg @ w + b

    # 均方误差损失
    loss = np.mean((y_pred - y_reg) ** 2)

    loss_history.append(loss)

    # 梯度计算
    dw = (2 / len(X_reg)) * X_reg.T @ (y_pred - y_reg)
    db = (2 / len(X_reg)) * np.sum(y_pred - y_reg)

    # 梯度下降
    w = w - lr * dw
    b = b - lr * db

# 输出结果
print("\n线性回归训练完成")
print(f"最终权重 w = {w[0][0]:.4f}")
print(f"最终偏置 b = {b:.4f}")

print(f"\n拟合直线方程：")
print(f"y = {w[0][0]:.4f}x + {b:.4f}")

# =========================================================
# 绘制线性回归损失下降曲线
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("线性回归损失下降曲线")
plt.grid()

# 保存图片
plt.savefig("线性回归损失下降曲线.png")

# =========================================================
# 绘制线性回归拟合效果图
# =========================================================

plt.figure(figsize=(8, 5))

# 散点图
plt.scatter(X_reg, y_reg, label='真实数据')

# 拟合直线
x_line = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_line = x_line @ w + b

plt.plot(x_line, y_line, color='red', label='拟合直线')

plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归拟合效果图")

plt.legend()
plt.grid()

# 保存图片
plt.savefig("线性回归拟合图.png")

# =========================================================
# 二、逻辑回归手动实现
# =========================================================

print("\n")
print("=" * 50)
print("逻辑回归实验开始")
print("=" * 50)

# 生成分类数据
X_cls, y_cls = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42
)

# 转换维度
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 参数初始化
w_log = np.random.randn(2, 1)
b_log = 0

# 超参数
lr_log = 0.1
epochs_log = 1000

# 保存损失
loss_history_log = []

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 开始训练
for epoch in range(epochs_log):

    # 前向传播
    z = X_train @ w_log + b_log

    y_pred = sigmoid(z)

    # 交叉熵损失
    loss = -np.mean(
        y_train * np.log(y_pred + 1e-8) +
        (1 - y_train) * np.log(1 - y_pred + 1e-8)
    )

    loss_history_log.append(loss)

    # 梯度
    dw = (1 / len(X_train)) * X_train.T @ (y_pred - y_train)

    db = (1 / len(X_train)) * np.sum(y_pred - y_train)

    # 参数更新
    w_log = w_log - lr_log * dw

    b_log = b_log - lr_log * db

# =========================================================
# 测试集准确率
# =========================================================

y_test_pred = sigmoid(X_test @ w_log + b_log)

y_test_pred = (y_test_pred >= 0.5).astype(int)

accuracy = np.mean(y_test_pred == y_test)

print("\n逻辑回归训练完成")
print(f"测试集准确率：{accuracy * 100:.2f}%")

# =========================================================
# 绘制逻辑回归损失下降曲线
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history_log)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("逻辑回归损失下降曲线")

plt.grid()

# 保存图片
plt.savefig("逻辑回归损失下降曲线.png")

# =========================================================
# 绘制逻辑回归决策边界
# =========================================================

plt.figure(figsize=(8, 5))

# 数据散点图
plt.scatter(
    X_cls[:, 0],
    X_cls[:, 1],
    c=y_cls,
    cmap='bwr'
)

# 决策边界
x1 = np.linspace(X_cls[:, 0].min(), X_cls[:, 0].max(), 100)

x2 = -(w_log[0] * x1 + b_log) / w_log[1]

plt.plot(x1, x2, color='black', label='决策边界')

plt.xlabel("特征1")
plt.ylabel("特征2")

plt.title("逻辑回归决策边界图")

plt.legend()
plt.grid()

# 保存图片
plt.savefig("逻辑回归决策边界图.png")

# =========================================================
# 三、学习率影响实验
# =========================================================

print("\n")
print("=" * 50)
print("学习率影响实验开始")
print("=" * 50)

learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(8, 5))

for lr_test in learning_rates:

    # 参数初始化
    w_lr = np.random.randn(1, 1)
    b_lr = 0

    losses = []

    # 训练
    for epoch in range(500):

        y_pred = X_reg @ w_lr + b_lr

        loss = np.mean((y_pred - y_reg) ** 2)

        losses.append(loss)

        # 梯度
        dw = (2 / len(X_reg)) * X_reg.T @ (y_pred - y_reg)

        db = (2 / len(X_reg)) * np.sum(y_pred - y_reg)

        # 参数更新
        w_lr = w_lr - lr_test * dw

        b_lr = b_lr - lr_test * db

    # 绘图
    plt.plot(losses, label=f"lr={lr_test}")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("不同学习率损失下降对比图")

plt.legend()
plt.grid()

# 保存图片
plt.savefig("不同学习率损失对比图.png")

# =========================================================
# 显示所有图片
# =========================================================

plt.show()

# =========================================================
# 实验总结输出
# =========================================================

print("\n")
print("=" * 50)
print("实验全部完成")
print("=" * 50)

print("""
已生成如下实验结果图片：

1. 线性回归损失下降曲线.png
2. 线性回归拟合图.png
3. 逻辑回归损失下降曲线.png
4. 逻辑回归决策边界图.png
5. 不同学习率损失对比图.png

可直接插入实验报告。
""")