# -*- coding: utf-8 -*-
"""
三层神经网络实现手写数字分类（MNIST）
- 仅依赖 numpy + matplotlib
- 手写三层全连接网络(含ReLU/Dropout)、反向传播、Adam优化
- 训练(train):
    * 打印每轮loss/acc
    * 保存最佳模型到 checkpoints/mlp3_best.npz
    * 生成训练过程可视化图:
        - training_loss_curve.png
        - training_acc_curve.png
- 推断(infer):
    * 加载最佳模型
    * 打印测试集最终准确率
    * 打印前10个预测结果
    * 生成 infer_examples.png (9张手写数字+预测结果)
- 交互模式:
    程序启动后会问你想跑 train 还是 infer
"""

import os
import time
import gzip
import struct
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 数据加载模块
# =========================

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dst_path: str):
    """
    如果本地没这个.gz文件，就下载。
    如果学校网络访问不了 Google，这一步可能会报错，
    你可以手动把4个.gz放到 data/ 目录再跑。
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        print(f"[Data] Downloading {url} -> {dst_path}")
        urllib.request.urlretrieve(url, dst_path)


def _read_images_gz(path: str) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number for images in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
    return data


def _read_labels_gz(path: str) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number for labels in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir: str = "data"):
    """
    返回:
    X_train: [N,784] float32 (0~1)
    y_train: [N] int64
    X_test : [N,784] float32 (0~1)
    y_test : [N] int64
    """
    os.makedirs(data_dir, exist_ok=True)

    paths = {
        k: os.path.join(data_dir, os.path.basename(v))
        for k, v in MNIST_URLS.items()
    }

    # 尝试下载（如果已存在则跳过）
    for key, url in MNIST_URLS.items():
        try:
            _download(url, paths[key])
        except Exception as e:
            print(f"[Warn] Download failed: {url}")
            print(e)
            print("请手动把四个MNIST .gz文件放到 data/ 目录后重试。")

    X_train = _read_images_gz(paths["train_images"]).astype(np.float32) / 255.0
    y_train = _read_labels_gz(paths["train_labels"]).astype(np.int64)
    X_test = _read_images_gz(paths["test_images"]).astype(np.float32) / 255.0
    y_test = _read_labels_gz(paths["test_labels"]).astype(np.int64)

    assert X_train.shape[1] == 784 and X_test.shape[1] == 784
    return X_train, y_train, X_test, y_test


# =========================
# 2. 基本单元模块
# =========================

class Linear:
    """
    全连接层 y = xW + b
    - forward缓存输入
    - backward计算dW/db/dX
    权重He初始化: N(0, sqrt(2/fan_in))
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        N = self._x.shape[0]
        self.dW = (self._x.T @ grad_out) / N
        self.db = np.mean(grad_out, axis=0, keepdims=True)
        grad_x = grad_out @ self.W.T
        return grad_x

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)


class ReLU:
    """ReLU(x)=max(0,x)"""
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(np.float32)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Dropout:
    """
    训练时随机丢一部分神经元输出来防过拟合
    推断时不丢弃
    """
    def __init__(self, p: float = 0.0, seed: int = 123):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = np.random.default_rng(seed)
        self._mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep_prob).astype(np.float32) / keep_prob
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return grad_out
        return grad_out * self._mask


class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵 (数值稳定)
    forward(logits,y_true):
      logits: [N,C]
      y_true: [N] int标签 或 [N,C] one-hot
    返回:
      loss(标量), grad_logits([N,C])
    """
    @staticmethod
    def forward(logits: np.ndarray, y_true: np.ndarray):
        N, C = logits.shape

        # one-hot
        if y_true.ndim == 1:
            y_oh = np.zeros((N, C), dtype=np.float32)
            y_oh[np.arange(N), y_true] = 1.0
        else:
            y_oh = y_true.astype(np.float32)

        # 稳定softmax
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        eps = 1e-12
        log_probs = np.log(probs + eps)
        loss = -np.sum(y_oh * log_probs) / N

        grad_logits = (probs - y_oh) / N
        return float(loss), grad_logits


# =========================
# 3. 网络结构模块 (784 -> 200 -> 100 -> 10)
# =========================

class MLP3:
    """
    三层全连接网络:
      输入784 -> 隐藏层1(h1) -> 隐藏层2(h2) -> 输出10类
    隐藏层后：ReLU + Dropout(训练时开启)
    """
    def __init__(self,
                 input_dim=784,
                 h1=200,
                 h2=100,
                 num_classes=10,
                 dropout_p=0.1,
                 seed=2025):

        self.fc1 = Linear(input_dim, h1, seed=seed + 1)
        self.relu1 = ReLU()
        self.do1 = Dropout(dropout_p, seed=seed + 11)

        self.fc2 = Linear(h1, h2, seed=seed + 2)
        self.relu2 = ReLU()
        self.do2 = Dropout(dropout_p, seed=seed + 22)

        self.fc3 = Linear(h2, num_classes, seed=seed + 3)

        self.criterion = SoftmaxCrossEntropy()
        self.training = True

    def train(self):
        self.training = True
        self.do1.training = True
        self.do2.training = True

    def eval(self):
        self.training = False
        self.do1.training = False
        self.do2.training = False

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()
        self.fc3.zero_grad()

    def forward(self, X: np.ndarray):
        z1 = self.fc1.forward(X)
        a1 = self.relu1.forward(z1)
        a1 = self.do1.forward(a1)

        z2 = self.fc2.forward(a1)
        a2 = self.relu2.forward(z2)
        a2 = self.do2.forward(a2)

        logits = self.fc3.forward(a2)
        return logits

    def loss_and_backward(self, X: np.ndarray, y: np.ndarray):
        logits = self.forward(X)
        loss, grad_logits = self.criterion.forward(logits, y)

        grad = self.fc3.backward(grad_logits)
        grad = self.do2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.do1.backward(grad)
        grad = self.relu1.backward(grad)
        _ = self.fc1.backward(grad)

        return loss

    def predict(self, X: np.ndarray):
        self.eval()
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def parameters(self):
        return [
            {"W": self.fc1.W, "b": self.fc1.b, "dW": self.fc1.dW, "db": self.fc1.db},
            {"W": self.fc2.W, "b": self.fc2.b, "dW": self.fc2.dW, "db": self.fc2.db},
            {"W": self.fc3.W, "b": self.fc3.b, "dW": self.fc3.dW, "db": self.fc3.db},
        ]


# =========================
# 4. 优化器 (Adam)
# =========================

class Adam:
    """
    Adam优化器 (带L2正则)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.mW = [np.zeros_like(p["W"]) for p in params]
        self.vW = [np.zeros_like(p["W"]) for p in params]
        self.mb = [np.zeros_like(p["b"]) for p in params]
        self.vb = [np.zeros_like(p["b"]) for p in params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            dW = p["dW"] + self.weight_decay * p["W"]
            db = p["db"]

            self.mW[i] = self.b1 * self.mW[i] + (1 - self.b1) * dW
            self.vW[i] = self.b2 * self.vW[i] + (1 - self.b2) * (dW * dW)
            self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * db
            self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * (db * db)

            mW_hat = self.mW[i] / (1 - self.b1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.b2 ** self.t)
            mb_hat = self.mb[i] / (1 - self.b1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.b2 ** self.t)

            p["W"] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            p["b"] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# =========================
# 5. 训练 / 推断 / 可视化
# =========================

def iterate_minibatches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


def accuracy(pred, y):
    return float((pred == y).mean())


def save_params(params, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{
        f"W{i}": p["W"] for i, p in enumerate(params)
    }, **{
        f"b{i}": p["b"] for i, p in enumerate(params)
    })
    print(f"[Save] parameters -> {path}")


def load_params(params, path: str):
    d = np.load(path)
    for i, p in enumerate(params):
        p["W"][:] = d[f"W{i}"]
        p["b"][:] = d[f"b{i}"]
    print(f"[Load] parameters <- {path}")


def evaluate(model: MLP3, X, y, batch_size=256) -> float:
    model.eval()
    preds = []
    for Xb, _ in iterate_minibatches(X, y, batch_size, shuffle=False):
        preds.append(model.predict(Xb))
    preds = np.concatenate(preds, axis=0)
    return accuracy(preds, y)


def train():
    """
    训练流程：
    - 加载数据
    - 构建模型和优化器
    - 多个epoch循环：前向/反向/更新
    - 记录loss和acc，保存最优模型
    - 生成训练曲线图（PNG）
    """
    data_dir = "data"
    ckpt_dir = "checkpoints"
    ckpt_path = os.path.join(ckpt_dir, "mlp3_best.npz")

    batch_size = 64          # 降一点峰值负载
    max_epochs = 12          # 不用太久，够收敛就行
    early_stop_patience = 3  # 早停
    lr = 1e-3
    weight_decay = 1e-4
    dropout_p = 0.1

    print("[INFO] Loading MNIST ...")
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    model = MLP3(784, 200, 100, 10, dropout_p=dropout_p, seed=2025)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    patience_cnt = 0

    hist_loss = []
    hist_train_acc = []
    hist_test_acc = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            model.zero_grad()
            loss = model.loss_and_backward(Xb, yb)
            optimizer.step()
            epoch_losses.append(loss)

        avg_loss = float(np.mean(epoch_losses))
        train_acc_v = evaluate(model, X_train, y_train, batch_size=256)
        test_acc_v = evaluate(model, X_test, y_test, batch_size=256)

        hist_loss.append(avg_loss)
        hist_train_acc.append(train_acc_v)
        hist_test_acc.append(test_acc_v)

        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}] "
              f"loss={avg_loss:.4f} | "
              f"train_acc={train_acc_v*100:.2f}% | "
              f"test_acc={test_acc_v*100:.2f}% | "
              f"time={dt:.1f}s")

        # 保存最优
        if test_acc_v > best_acc:
            best_acc = test_acc_v
            save_params(model.parameters(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop_patience:
                print(f"[EarlyStop] no improvement for {early_stop_patience} epochs.")
                break

    print(f"[DONE] Best Test Acc = {best_acc*100:.2f}%")
    print(f"[DONE] Best params saved at {ckpt_path}")

    # ----------- 画训练曲线并保存 -----------
    epochs_axis = range(1, len(hist_loss) + 1)

    # Loss曲线
    plt.figure()
    plt.plot(epochs_axis, hist_loss, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_curve.png", dpi=200)
    plt.close()

    # 准确率曲线
    plt.figure()
    plt.plot(epochs_axis,
             [a * 100 for a in hist_train_acc],
             label="train_acc(%)")
    plt.plot(epochs_axis,
             [a * 100 for a in hist_test_acc],
             label="test_acc(%)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_acc_curve.png", dpi=200)
    plt.close()

    print("[PLOT] Saved training_loss_curve.png and training_acc_curve.png")
    print("请把这两张图放进报告，说明loss下降、准确率上升。")


def infer():
    """
    推断流程：
    - 加载MNIST测试集
    - 从checkpoints加载最优模型
    - 打印整体测试准确率
    - 打印前10条预测结果
    - 画前9张测试图 + 模型预测，保存infer_examples.png
    """
    data_dir = "data"
    ckpt_path = os.path.join("checkpoints", "mlp3_best.npz")

    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    # 推断时不需要Dropout
    model = MLP3(784, 200, 100, 10, dropout_p=0.0, seed=2025)
    load_params(model.parameters(), ckpt_path)

    pred = model.predict(X_test)
    acc = accuracy(pred, y_test)
    print(f"[Infer] Test Acc = {acc*100:.2f}%")

    for i in range(10):
        print(f"idx={i:02d} | gt={y_test[i]} | pred={pred[i]} | correct={y_test[i]==pred[i]}")

    # 可视化前9张预测
    plt.figure(figsize=(6, 6))
    for i in range(9):
        img = X_test[i].reshape(28, 28)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"gt:{y_test[i]} pred:{pred[i]}", fontsize=9)
    plt.suptitle(f"MNIST predictions (acc={acc*100:.2f}%)")
    plt.tight_layout()
    plt.savefig("infer_examples.png", dpi=200)
    plt.close()

    print("[PLOT] Saved infer_examples.png")
    print("请把 infer_examples.png 放到报告里证明推断效果。")


# =========================
# 6. 主入口（交互模式）
# =========================

if __name__ == "__main__":
    """
    运行方式：
    - 直接点运行（Run）即可
    - 程序会询问你选择 train 还是 infer
    - 你输入 train 回车 = 训练并画训练曲线
    - 你输入 infer 回车 = 加载最优模型并可视化推断效果
    """

    mode = input("请选择模式 (train / infer): ").strip().lower()

    if mode == "train":
        train()
    elif mode == "infer":
        infer()
    else:
        print("未识别的模式，请重新运行程序并输入 train 或 infer")
