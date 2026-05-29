"""
实验2：基于 NumPy 的浅层神经网络实现
包含：前向传播、反向传播、训练评估、决策边界可视化、与逻辑回归对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 一、数据集
# ═══════════════════════════════════════════════════════════════
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════
# 二、浅层神经网络类（NumPy 手动实现）
# ═══════════════════════════════════════════════════════════════
class ShallowNN:
    """
    结构：输入(2) → 隐藏层(hidden_size, ReLU/Tanh) → 输出(1, Sigmoid)
    损失：二元交叉熵
    优化：批量梯度下降
    """

    def __init__(self, hidden_size=8, activation='relu', lr=0.05, seed=0):
        self.hidden_size = hidden_size
        self.activation  = activation
        self.lr          = lr
        rng = np.random.default_rng(seed)

        # He 初始化（适合 ReLU），Xavier 适合 Tanh
        if activation == 'relu':
            scale1 = np.sqrt(2.0 / 2)
        else:
            scale1 = np.sqrt(1.0 / 2)
        scale2 = np.sqrt(1.0 / hidden_size)

        self.W1 = rng.normal(0, scale1, (2, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, scale2, (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    # ── 激活函数 ──────────────────────────────────────────────
    def _act(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        return np.tanh(z)

    def _act_deriv(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # ── 前向传播 ──────────────────────────────────────────────
    def forward(self, X):
        self.X   = X
        self.Z1  = X @ self.W1 + self.b1        # (n, hidden)
        self.A1  = self._act(self.Z1)            # (n, hidden)
        self.Z2  = self.A1 @ self.W2 + self.b2  # (n, 1)
        self.A2  = self._sigmoid(self.Z2)        # (n, 1) — 输出概率
        return self.A2

    # ── 损失函数 ──────────────────────────────────────────────
    def loss(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        return -np.mean(y_true * np.log(y_pred + 1e-9) +
                        (1 - y_true) * np.log(1 - y_pred + 1e-9))

    # ── 反向传播 ──────────────────────────────────────────────
    def backward(self, y_true):
        n = self.X.shape[0]
        y_true = y_true.reshape(-1, 1)

        # 输出层误差（交叉熵 + Sigmoid 联合导数）
        dZ2 = self.A2 - y_true                   # (n, 1)

        # 输出层参数梯度
        dW2 = self.A1.T @ dZ2 / n               # (hidden, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n  # (1, 1)

        # 将误差传回隐藏层（链式法则）
        dA1 = dZ2 @ self.W2.T                   # (n, hidden)
        dZ1 = dA1 * self._act_deriv(self.Z1)    # (n, hidden)

        # 隐藏层参数梯度
        dW1 = self.X.T @ dZ1 / n               # (2, hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n  # (1, hidden)

        # 参数更新
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── 训练 ──────────────────────────────────────────────────
    def train(self, X_tr, y_tr, X_te, y_te, epochs=3000):
        loss_hist, acc_hist = [], []
        for _ in range(epochs):
            y_pred = self.forward(X_tr)
            loss_hist.append(self.loss(y_tr, y_pred))
            self.backward(y_tr)
            # 测试准确率
            y_te_pred = (self.forward(X_te) >= 0.5).astype(int).flatten()
            acc_hist.append(np.mean(y_te_pred == y_te))
        return loss_hist, acc_hist

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()


# ═══════════════════════════════════════════════════════════════
# 三、决策边界绘制工具
# ═══════════════════════════════════════════════════════════════
def plot_decision_boundary(model, X, y, ax, title='', show_points=True):
    x1_min, x1_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    x2_min, x2_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                            np.linspace(x2_min, x2_max, 300))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict_proba(grid).reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, levels=50, cmap='RdBu', alpha=0.4, vmin=0, vmax=1)
    ax.contour(xx1, xx2, Z, levels=[0.5], colors='k', linewidths=2)
    if show_points:
        for cls, color, marker in [(0, 'steelblue', 'o'), (1, 'tomato', '^')]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                       s=25, alpha=0.8, edgecolors='white', linewidths=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(x1_min, x1_max); ax.set_ylim(x2_min, x2_max)
    ax.set_xticks([]); ax.set_yticks([])


# ═══════════════════════════════════════════════════════════════
# 四、训练主模型（hidden=8, relu）并记录曲线
# ═══════════════════════════════════════════════════════════════
print("\n训练主模型 (hidden=8, relu)...")
main_model = ShallowNN(hidden_size=8, activation='relu', lr=0.05)
loss_hist, acc_hist = main_model.train(X_train, y_train, X_test, y_test, epochs=3000)
final_acc = acc_hist[-1]
print(f"  最终测试准确率: {final_acc*100:.2f}%")

# ── 图1：训练损失 + 测试准确率曲线 ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_axis = range(1, 3001)

ax1.plot(epochs_axis, loss_hist, color='steelblue', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_title('Training Loss Curve (hidden=8, ReLU)', fontsize=13)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_axis, [a*100 for a in acc_hist], color='darkorange', linewidth=1.5)
ax2.axhline(y=final_acc*100, color='red', linestyle='--', linewidth=1,
            label=f'Final: {final_acc*100:.1f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Test Accuracy Curve (hidden=8, ReLU)', fontsize=13)
ax2.set_ylim(50, 102)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_train_curves.png', dpi=150)
plt.close()
print("Saved: fig1_train_curves.png")


# ═══════════════════════════════════════════════════════════════
# 五、不同隐藏层大小的决策边界对比
# ═══════════════════════════════════════════════════════════════
hidden_sizes = [2, 4, 8, 16, 32]
print("\n训练不同隐藏层大小的模型...")

models = {}
accs   = {}
for hs in hidden_sizes:
    m = ShallowNN(hidden_size=hs, activation='relu', lr=0.05, seed=42)
    _, acc_h = m.train(X_train, y_train, X_test, y_test, epochs=3000)
    models[hs] = m
    accs[hs]   = acc_h[-1]
    print(f"  hidden={hs:2d}  test_acc={accs[hs]*100:.2f}%")

# ── 图2：5种隐藏层大小的决策边界对比 ─────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
for ax, hs in zip(axes, hidden_sizes):
    plot_decision_boundary(models[hs], X, y, ax,
                           title=f'Hidden={hs}\nAcc={accs[hs]*100:.1f}%')
plt.suptitle('Decision Boundary vs. Hidden Layer Size (ReLU, 3000 epochs)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_hidden_size_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_hidden_size_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 六、选做：与逻辑回归对比
# ═══════════════════════════════════════════════════════════════
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=3000):
        self.lr, self.epochs = lr, epochs
        self.W = np.zeros(2)
        self.b = 0.0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            yh = self.sigmoid(X @ self.W + self.b)
            e = yh - y
            self.W -= self.lr / n * (X.T @ e)
            self.b -= self.lr / n * e.sum()

    def predict_proba(self, X):
        return self.sigmoid(X @ self.W + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


print("\n训练逻辑回归对比模型...")
lr_model = LogisticRegression(lr=0.1, epochs=3000)
lr_model.fit(X_train, y_train)
lr_acc = np.mean(lr_model.predict(X_test) == y_test)
print(f"  逻辑回归测试准确率: {lr_acc*100:.2f}%")
print(f"  神经网络测试准确率: {final_acc*100:.2f}%")

# ── 图3：逻辑回归 vs 神经网络决策边界 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(lr_model, X, y, axes[0],
                       title=f'Logistic Regression\nTest Acc: {lr_acc*100:.1f}%')
plot_decision_boundary(main_model, X, y, axes[1],
                       title=f'Neural Network (hidden=8)\nTest Acc: {final_acc*100:.1f}%')
plt.suptitle('Logistic Regression vs. Neural Network — Decision Boundary',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_lr_vs_nn.png', dpi=150)
plt.close()
print("Saved: fig3_lr_vs_nn.png")

print("\n=== 所有图片已保存至", OUTPUT_DIR, "===")
print(f"最佳隐藏层大小: {max(accs, key=accs.get)}, 准确率: {max(accs.values())*100:.2f}%")
