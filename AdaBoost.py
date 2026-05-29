import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class DecisionStump:
    """决策树桩（单层决策树）作为弱学习器"""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1  # 1表示小于阈值为-1，大于为1；-1则相反
        self.alpha = None

    def fit(self, X, y, sample_weight):
        """训练决策树桩"""
        n_samples, n_features = X.shape
        min_error = float('inf')

        # 遍历所有特征
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            unique_values = np.unique(feature_values)

            # 遍历所有可能的阈值
            for threshold in unique_values:
                # 尝试两种极性
                for polarity in [1, -1]:
                    predictions = self._predict_with_params(X, feature_i, threshold, polarity)

                    # 计算加权错误率
                    error = np.sum(sample_weight[predictions != y])

                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_i
                        self.threshold = threshold
                        self.polarity = polarity

        return min_error

    def _predict_with_params(self, X, feature_index, threshold, polarity):
        """根据给定参数进行预测"""
        predictions = np.ones(X.shape[0])
        if polarity == 1:
            predictions[X[:, feature_index] < threshold] = -1
        else:
            predictions[X[:, feature_index] < threshold] = 1
        return predictions

    def predict(self, X):
        """预测"""
        return self._predict_with_params(X, self.feature_index, self.threshold, self.polarity)


class AdaBoost:
    """AdaBoost算法实现"""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        """训练AdaBoost模型"""
        n_samples = X.shape[0]

        # 初始化样本权重
        sample_weight = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # 创建弱学习器
            stump = DecisionStump()

            # 训练弱学习器
            error = stump.fit(X, y, sample_weight)

            # 如果错误率为0或大于0.5，停止训练
            if error == 0 or error >= 0.5:
                if error == 0:
                    stump.alpha = 10  # 给完美分类器一个大权重
                    self.estimators.append(stump)
                    self.alphas.append(stump.alpha)
                break

            # 计算弱学习器权重
            alpha = 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha

            # 保存弱学习器
            self.estimators.append(stump)
            self.alphas.append(alpha)

            # 获取弱学习器预测
            predictions = stump.predict(X)

            # 更新样本权重
            sample_weight *= np.exp(-alpha * y * predictions)
            sample_weight /= np.sum(sample_weight)  # 归一化

            print(f"第{i + 1}个基学习器训练完成，错误率: {error:.4f}, 权重: {alpha:.4f}")

    def predict(self, X):
        """预测"""
        predictions = np.zeros(X.shape[0])

        for estimator, alpha in zip(self.estimators, self.alphas):
            predictions += alpha * estimator.predict(X)

        return np.sign(predictions)

    def staged_predict(self, X):
        """逐步预测，返回每个阶段的预测结果"""
        predictions = np.zeros(X.shape[0])
        staged_predictions = []

        for estimator, alpha in zip(self.estimators, self.alphas):
            predictions += alpha * estimator.predict(X)
            staged_predictions.append(np.sign(predictions.copy()))

        return staged_predictions


# 西瓜数据集3.0α
def load_watermelon_data():
    """加载西瓜数据集3.0α"""
    data = {
        '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657,
                 0.360, 0.593, 0.719],
        '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198,
                   0.370, 0.042, 0.103],
        '好瓜': [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    }

    df = pd.DataFrame(data)
    X = df[['密度', '含糖率']].values
    y = df['好瓜'].values

    return X, y


def plot_decision_boundary(X, y, clf, n_estimators_list=[3, 5, 11], title="AdaBoost分类边界"):
    """绘制决策边界"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 创建网格点
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 获取各阶段预测结果
    staged_predictions = clf.staged_predict(grid_points)

    for idx, n_est in enumerate(n_estimators_list):
        if n_est <= len(staged_predictions):
            ax = axes[idx]

            # 获取该阶段的预测结果
            Z = staged_predictions[n_est - 1].reshape(xx.shape)

            # 绘制决策边界
            ax.contourf(xx, yy, Z, levels=[-1.5, -0.5, 0.5, 1.5],
                        colors=['lightblue', 'lightcoral'], alpha=0.6)
            ax.contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)

            # 绘制数据点
            good_melon = y == 1
            bad_melon = y == -1

            ax.scatter(X[good_melon, 0], X[good_melon, 1],
                       c='green', marker='+', s=100, label='好瓜', linewidths=2)
            ax.scatter(X[bad_melon, 0], X[bad_melon, 1],
                       c='red', marker='_', s=100, label='坏瓜', linewidths=2)

            # 绘制基学习器的分割线
            for i in range(min(n_est, len(clf.estimators))):
                stump = clf.estimators[i]
                if stump.feature_index == 0:  # 密度特征
                    ax.axvline(x=stump.threshold, color='black', linestyle='--', alpha=0.7)
                else:  # 含糖率特征
                    ax.axhline(y=stump.threshold, color='black', linestyle='--', alpha=0.7)

            ax.set_xlabel('密度')
            ax.set_ylabel('含糖率')
            ax.set_title(f'({chr(97 + idx)}) {n_est}个基学习器')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("AdaBoost算法在西瓜数据集3.0α上的应用")
    print("=" * 50)

    # 加载数据
    X, y = load_watermelon_data()
    print(f"数据集大小: {X.shape}")
    print(f"正样本数量: {np.sum(y == 1)}")
    print(f"负样本数量: {np.sum(y == -1)}")
    print()

    # 训练AdaBoost模型
    print("开始训练AdaBoost模型...")
    clf = AdaBoost(n_estimators=11)
    clf.fit(X, y)
    print()

    # 评估模型性能
    train_pred = clf.predict(X)
    train_accuracy = np.mean(train_pred == y)
    print(f"训练集准确率: {train_accuracy:.4f}")

    # 输出各个基学习器信息
    print("\n基学习器信息:")
    for i, (estimator, alpha) in enumerate(zip(clf.estimators, clf.alphas)):
        feature_name = "密度" if estimator.feature_index == 0 else "含糖率"
        print(f"基学习器{i + 1}: 特征={feature_name}, 阈值={estimator.threshold:.3f}, "
              f"极性={estimator.polarity}, 权重={alpha:.4f}")

    # 绘制决策边界
    print("\n绘制决策边界...")
    plot_decision_boundary(X, y, clf)

    # 分析各阶段性能
    print("\n各阶段性能分析:")
    staged_predictions = clf.staged_predict(X)
    for i, pred in enumerate(staged_predictions):
        accuracy = np.mean(pred == y)
        print(f"{i + 1}个基学习器: 准确率 = {accuracy:.4f}")


if __name__ == "__main__":
    main()