import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DecisionTreeNode:
    """决策树节点类"""
    def __init__(self):
        self.feature = None       # 分割特征
        self.threshold = None     # 分割阈值
        self.left = None          # 左子树
        self.right = None         # 右子树
        self.prediction = None    # 叶节点预测值
        self.samples = 0          # 样本数
        self.gini = 0            # 基尼不纯度

class DecisionTreeClassifier:
    """决策树分类器"""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_importances_ = None
        self.feature_names = None
        self.class_names = None

    def calculate_gini(self, y):
        """计算基尼不纯度"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def find_best_split(self, X, y):
        """寻找最佳分割"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]

                left_gini = self.calculate_gini(left_y)
                right_gini = self.calculate_gini(right_y)

                weighted_gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def build_tree(self, X, y, depth=0):
        """构建决策树"""
        node = DecisionTreeNode()
        node.samples = len(y)
        node.gini = self.calculate_gini(y)

        # 检查停止条件
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            node.prediction = unique_classes[0]
            return node

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # 选择最多的类作为预测
            node.prediction = np.bincount(y).argmax()
            return node

        # 寻找最佳分割
        best_feature, best_threshold, best_gini = self.find_best_split(X, y)

        if best_feature is None:
            node.prediction = np.bincount(y).argmax()
            return node

        node.feature = best_feature
        node.threshold = best_threshold

        # 分割数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        # 递归构建子树
        node.left = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return node

    def fit(self, X, y, feature_names=None, class_names=None):
        """训练模型"""
        self.feature_names = feature_names
        self.class_names = class_names
        self.root = self.build_tree(X, y)
        self.calculate_feature_importances(X, y)

    def predict_sample(self, node, sample):
        """预测单个样本"""
        if node.prediction is not None:
            return node.prediction

        if sample[node.feature] <= node.threshold:
            return self.predict_sample(node.left, sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict(self, X):
        """预测"""
        return np.array([self.predict_sample(self.root, sample) for sample in X])

    def calculate_feature_importances(self, X, y):
        """计算特征重要性"""
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        self.calculate_node_importance(self.root, len(y))

        # 归一化
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

    def calculate_node_importance(self, node, total_samples):
        """计算节点重要性"""
        if node.prediction is not None:
            return

        importance = (node.samples / total_samples) * node.gini
        self.feature_importances_[node.feature] += importance

        if node.left:
            self.calculate_node_importance(node.left, total_samples)
        if node.right:
            self.calculate_node_importance(node.right, total_samples)

    def get_tree_text(self, node=None, depth=0, prefix=""):
        """获取树的文本表示"""
        if node is None:
            node = self.root
        if node is None:
            return "空树"

        result = ""
        indent = "  " * depth

        if node.prediction is not None:
            class_name = self.class_names[node.prediction] if self.class_names else str(node.prediction)
            result += f"{prefix}预测: {class_name} (样本数: {node.samples}, 基尼: {node.gini:.3f})\n"
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names else f"特征{node.feature}"
            result += f"{prefix}{feature_name} <= {node.threshold:.2f} (样本数: {node.samples}, 基尼: {node.gini:.3f})\n"
            if node.left:
                result += self.get_tree_text(node.left, depth + 1, indent + "├─ 是: ")
            if node.right:
                result += self.get_tree_text(node.right, depth + 1, indent + "└─ 否: ")

        return result

class IrisClassificationProject:
    """鸢尾花分类项目主类"""
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
        self.class_names = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

    def load_data(self):
        """加载鸢尾花数据集"""
        iris = load_iris()
        return iris.data, iris.target

    def prepare_data(self, test_size=0.3, random_state=42):
        """准备数据"""
        X, y = self.load_data()

        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"训练集样本数: {len(self.X_train)}")
        print(f"测试集样本数: {len(self.X_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, max_depth=5, min_samples_split=2):
        """训练模型"""
        print("开始训练决策树模型...")

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        self.model.fit(
            self.X_train,
            self.y_train,
            feature_names=self.feature_names,
            class_names=self.class_names
        )

        print("模型训练完成!")

    def evaluate_model(self):
        """评估模型"""
        # 训练集预测
        y_train_pred = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        # 测试集预测
        self.y_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, self.y_pred)

        print(f"\n模型性能评估:")
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        # 详细分类报告
        print(f"\n详细分类报告:")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.class_names,
            zero_division=0
        ))

        return train_accuracy, test_accuracy

    def plot_data_distribution(self):
        """绘制数据分布图"""
        X, y = self.load_data()
        df = pd.DataFrame(X, columns=self.feature_names)
        df['类别'] = [self.class_names[i] for i in y]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('鸢尾花数据集特征分布', fontsize=16, y=1.02)

        # 花萼长度 vs 花萼宽度
        sns.scatterplot(data=df, x='花萼长度', y='花萼宽度', hue='类别', ax=axes[0,0], s=60)
        axes[0,0].set_title('花萼长度 vs 花萼宽度')
        axes[0,0].grid(True, alpha=0.3)

        # 花瓣长度 vs 花瓣宽度
        sns.scatterplot(data=df, x='花瓣长度', y='花瓣宽度', hue='类别', ax=axes[0,1], s=60)
        axes[0,1].set_title('花瓣长度 vs 花瓣宽度')
        axes[0,1].grid(True, alpha=0.3)

        # 花萼长度 vs 花瓣长度
        sns.scatterplot(data=df, x='花萼长度', y='花瓣长度', hue='类别', ax=axes[1,0], s=60)
        axes[1,0].set_title('花萼长度 vs 花瓣长度')
        axes[1,0].grid(True, alpha=0.3)

        # 花萼宽度 vs 花瓣宽度
        sns.scatterplot(data=df, x='花萼宽度', y='花瓣宽度', hue='类别', ax=axes[1,1], s=60)
        axes[1,1].set_title('花萼宽度 vs 花瓣宽度')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        if self.model is None or self.model.feature_importances_ is None:
            print("请先训练模型!")
            return

        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=True)

        plt.barh(importance_df['特征'], importance_df['重要性'], color='skyblue', alpha=0.8)
        plt.xlabel('特征重要性')
        plt.title('决策树特征重要性分析')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(importance_df['重要性']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        if self.y_pred is None:
            print("请先进行模型预测!")
            return

        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, feature_pair=(2, 3)):
        """绘制决策边界（选择两个特征）"""
        if self.model is None:
            print("请先训练模型!")
            return

        X, y = self.load_data()
        X_subset = X[:, feature_pair]

        # 创建简单的2D决策树
        model_2d = DecisionTreeClassifier(max_depth=3)
        model_2d.fit(X_subset, y,
                    feature_names=[self.feature_names[i] for i in feature_pair],
                    class_names=self.class_names)

        # 创建网格
        h = 0.02
        x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

        # 预测网格点
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(12, 8))

        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')

        # 绘制数据点
        colors = ['red', 'green', 'blue']
        for i, class_name in enumerate(self.class_names):
            idx = y == i
            plt.scatter(X_subset[idx, 0], X_subset[idx, 1],
                       c=colors[i], label=class_name, s=60, alpha=0.8)

        plt.xlabel(self.feature_names[feature_pair[0]])
        plt.ylabel(self.feature_names[feature_pair[1]])
        plt.title(f'决策边界 - {self.feature_names[feature_pair[0]]} vs {self.feature_names[feature_pair[1]]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_decision_tree(self):
        """打印决策树结构"""
        if self.model is None:
            print("请先训练模型!")
            return

        print("\n决策树结构:")
        print("=" * 50)
        print(self.model.get_tree_text())

    def predict_new_sample(self, sample):
        """预测新样本"""
        if self.model is None:
            print("请先训练模型!")
            return None

        prediction = self.model.predict([sample])[0]
        class_name = self.class_names[prediction]

        print(f"\n输入样本: {sample}")
        print(f"预测结果: {class_name} (类别 {prediction})")

        return prediction

    def run_complete_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print("          鸢尾花数据集分类项目")
        print("=" * 60)

        # 1. 数据准备
        print("\n1. 数据准备")
        print("-" * 30)
        self.prepare_data()

        # 2. 数据可视化
        print("\n2. 数据分布可视化")
        print("-" * 30)
        self.plot_data_distribution()

        # 3. 模型训练
        print("\n3. 模型训练")
        print("-" * 30)
        self.train_model()

        # 4. 模型评估
        print("\n4. 模型评估")
        print("-" * 30)
        self.evaluate_model()

        # 5. 结果可视化
        print("\n5. 结果可视化")
        print("-" * 30)
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        self.plot_decision_boundary()

        # 6. 决策树结构
        print("\n6. 决策树结构")
        print("-" * 30)
        self.print_decision_tree()

        # 7. 样本预测测试
        print("\n7. 样本预测测试")
        print("-" * 30)
        # 测试样本（山鸢尾典型特征）
        test_sample = [5.1, 3.5, 1.4, 0.2]
        self.predict_new_sample(test_sample)

        # 测试样本（变色鸢尾典型特征）
        test_sample = [6.0, 2.7, 4.0, 1.3]
        self.predict_new_sample(test_sample)

        # 测试样本（维吉尼亚鸢尾典型特征）
        test_sample = [6.5, 3.0, 5.8, 2.2]
        self.predict_new_sample(test_sample)

        print("\n" + "=" * 60)
        print("          分析完成!")
        print("=" * 60)

# 使用示例
if __name__ == "__main__":
    # 创建项目实例
    project = IrisClassificationProject()

    # 运行完整分析
    project.run_complete_analysis()

    # 也可以单独运行各个部分
    # project.prepare_data()
    # project.train_model()
    # project.evaluate_model()
    # project.plot_data_distribution()
    # project.plot_feature_importance()
    # project.plot_confusion_matrix()
    # project.plot_decision_boundary()
    # project.print_decision_tree()

    # 交互式预测
    print("\n交互式预测测试:")
    print("请输入鸢尾花的四个特征值（花萼长度 花萼宽度 花瓣长度 花瓣宽度）")
    print("示例: 5.1 3.5 1.4 0.2")

    try:
        user_input = input("请输入特征值（用空格分隔）: ")
        if user_input.strip():
            features = list(map(float, user_input.split()))
            if len(features) == 4:
                project.predict_new_sample(features)
            else:
                print("请输入4个特征值!")
    except:
        print("输入格式错误，请输入数字!")