# SVM算法在鸢尾花分类上的应用实验
# 作者：实验四
# 日期：2025年6月

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class IrisSVMClassifier:
    def __init__(self, data_path=None):
        """
        初始化SVM分类器
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.svm_model = None
        self.best_params = None

    def load_data(self, data_string=None):
        """
        加载鸢尾花数据集
        """
        if data_string:
            # 从字符串加载数据（用于演示）
            from io import StringIO
            self.data = pd.read_csv(StringIO(data_string))
        elif self.data_path:
            # 从文件路径加载数据
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("请提供数据路径或数据字符串")

        print("数据集基本信息：")
        print(f"数据集形状: {self.data.shape}")
        print(f"特征列: {list(self.data.columns[:-1])}")
        print(f"类别列: {self.data.columns[-1]}")
        print(f"类别分布:")
        print(self.data['class'].value_counts())

        return self.data

    def preprocess_data(self):
        """
        数据预处理
        """
        # 分离特征和标签
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values

        # 编码类别标签
        self.y = self.label_encoder.fit_transform(self.y)

        # 数据划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # 特征标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("数据预处理完成:")
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"类别编码: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

    def grid_search_optimization(self):
        """
        使用网格搜索优化SVM参数
        """
        # 定义参数网格
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }

        # 创建SVM模型
        svm = SVC(random_state=42)

        # 网格搜索
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )

        print("正在进行参数优化...")
        grid_search.fit(self.X_train_scaled, self.y_train)

        self.best_params = grid_search.best_params_
        self.svm_model = grid_search.best_estimator_

        print(f"最优参数: {self.best_params}")
        print(f"最优交叉验证得分: {grid_search.best_score_:.4f}")

        return self.best_params

    def train_model(self, use_grid_search=True):
        """
        训练SVM模型
        """
        if use_grid_search:
            self.grid_search_optimization()
        else:
            # 使用默认参数
            self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            self.svm_model.fit(self.X_train_scaled, self.y_train)

        # 交叉验证评估
        cv_scores = cross_val_score(self.svm_model, self.X_train_scaled, self.y_train, cv=5)
        print(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def evaluate_model(self):
        """
        评估模型性能
        """
        # 训练集预测
        y_train_pred = self.svm_model.predict(self.X_train_scaled)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        # 测试集预测
        y_test_pred = self.svm_model.predict(self.X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print("=" * 50)
        print("模型性能评估结果:")
        print("=" * 50)
        print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # 详细分类报告
        print("\n测试集详细分类报告:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_test_pred, target_names=target_names))

        return train_accuracy, test_accuracy, y_test_pred

    def plot_results(self):
        """
        可视化结果
        """
        # 预测结果
        y_test_pred = self.svm_model.predict(self.X_test_scaled)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 混淆矩阵
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测类别')
        axes[0, 0].set_ylabel('真实类别')

        # 2. 特征分布（前两个特征）
        colors = ['red', 'green', 'blue']
        class_names = self.label_encoder.classes_

        for i, class_name in enumerate(class_names):
            mask = self.y == i
            axes[0, 1].scatter(self.X[mask, 0], self.X[mask, 1],
                               c=colors[i], label=class_name, alpha=0.7)
        axes[0, 1].set_title('特征分布图 (花萼长度 vs 花萼宽度)')
        axes[0, 1].set_xlabel('花萼长度 (cm)')
        axes[0, 1].set_ylabel('花萼宽度 (cm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. PCA降维可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_test_scaled)

        for i, class_name in enumerate(class_names):
            mask = self.y_test == i
            axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                               c=colors[i], label=f'真实-{class_name}', alpha=0.7, s=50)

        # 添加预测错误的点
        wrong_predictions = self.y_test != y_test_pred
        if np.any(wrong_predictions):
            axes[1, 0].scatter(X_pca[wrong_predictions, 0], X_pca[wrong_predictions, 1],
                               c='black', marker='x', s=100, label='预测错误')

        axes[1, 0].set_title('PCA降维可视化 (测试集)')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 特征重要性（对于线性核）
        if self.svm_model.kernel == 'linear':
            feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
            # 线性SVM的特征权重
            coef = np.abs(self.svm_model.coef_[0])
            axes[1, 1].bar(feature_names, coef)
            axes[1, 1].set_title('特征重要性 (线性SVM权重)')
            axes[1, 1].set_ylabel('权重绝对值')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # 支持向量分布
            support_vector_counts = np.bincount(self.y_train[self.svm_model.support_])
            axes[1, 1].bar(range(len(support_vector_counts)), support_vector_counts,
                           color=colors[:len(support_vector_counts)])
            axes[1, 1].set_title('各类别支持向量数量')
            axes[1, 1].set_xlabel('类别')
            axes[1, 1].set_ylabel('支持向量数量')
            axes[1, 1].set_xticks(range(len(class_names)))
            axes[1, 1].set_xticklabels(class_names)

        plt.tight_layout()
        plt.show()

        # 打印支持向量信息
        print(f"\n支持向量信息:")
        print(f"支持向量总数: {len(self.svm_model.support_)}")
        print(f"各类别支持向量数: {np.bincount(self.y_train[self.svm_model.support_])}")

    def run_experiment(self, data_string=None):
        """
        运行完整实验
        """
        print("=" * 60)
        print("SVM算法在鸢尾花分类上的应用实验")
        print("=" * 60)

        # 1. 加载数据
        self.load_data(data_string)

        # 2. 数据预处理
        self.preprocess_data()

        # 3. 训练模型
        self.train_model(use_grid_search=True)

        # 4. 评估模型
        train_acc, test_acc, predictions = self.evaluate_model()

        # 5. 可视化结果
        self.plot_results()

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'best_params': self.best_params,
            'support_vectors': len(self.svm_model.support_),
            'model': self.svm_model
        }


# 主程序
if __name__ == "__main__":
    # 创建分类器实例
    classifier = IrisSVMClassifier()

    # 鸢尾花数据集（从提供的数据中读取）
    iris_data = """sepallength,sepalwidth,petallength,petalwidth,class
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3.0,1.4,0.1,setosa
4.3,3.0,1.1,0.1,setosa
5.8,4.0,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
5.4,3.4,1.7,0.2,setosa
5.1,3.7,1.5,0.4,setosa
4.6,3.6,1.0,0.2,setosa
5.1,3.3,1.7,0.5,setosa
4.8,3.4,1.9,0.2,setosa
5.0,3.0,1.6,0.2,setosa
5.0,3.4,1.6,0.4,setosa
5.2,3.5,1.5,0.2,setosa
5.2,3.4,1.4,0.2,setosa
4.7,3.2,1.6,0.2,setosa
4.8,3.1,1.6,0.2,setosa
5.4,3.4,1.5,0.4,setosa
5.2,4.1,1.5,0.1,setosa
5.5,4.2,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.0,3.2,1.2,0.2,setosa
5.5,3.5,1.3,0.2,setosa
4.9,3.1,1.5,0.1,setosa
4.4,3.0,1.3,0.2,setosa
5.1,3.4,1.5,0.2,setosa
5.0,3.5,1.3,0.3,setosa
4.5,2.3,1.3,0.3,setosa
4.4,3.2,1.3,0.2,setosa
5.0,3.5,1.6,0.6,setosa
5.1,3.8,1.9,0.4,setosa
4.8,3.0,1.4,0.3,setosa
5.1,3.8,1.6,0.2,setosa
4.6,3.2,1.4,0.2,setosa
5.3,3.7,1.5,0.2,setosa
5.0,3.3,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5.0,2.0,3.5,1.0,versicolor
5.9,3.0,4.2,1.5,versicolor
6.0,2.2,4.0,1.0,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3.0,4.5,1.5,versicolor
5.8,2.7,4.1,1.0,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor
5.9,3.2,4.8,1.8,versicolor
6.1,2.8,4.0,1.3,versicolor
6.3,2.5,4.9,1.5,versicolor
6.1,2.8,4.7,1.2,versicolor
6.4,2.9,4.3,1.3,versicolor
6.6,3.0,4.4,1.4,versicolor
6.8,2.8,4.8,1.4,versicolor
6.7,3.0,5.0,1.7,versicolor
6.0,2.9,4.5,1.5,versicolor
5.7,2.6,3.5,1.0,versicolor
5.5,2.4,3.8,1.1,versicolor
5.5,2.4,3.7,1.0,versicolor
5.8,2.7,3.9,1.2,versicolor
6.0,2.7,5.1,1.6,versicolor
5.4,3.0,4.5,1.5,versicolor
6.0,3.4,4.5,1.6,versicolor
6.7,3.1,4.7,1.5,versicolor
6.3,2.3,4.4,1.3,versicolor
5.6,3.0,4.1,1.3,versicolor
5.5,2.5,4.0,1.3,versicolor
5.5,2.6,4.4,1.2,versicolor
6.1,3.0,4.6,1.4,versicolor
5.8,2.6,4.0,1.2,versicolor
5.0,2.3,3.3,1.0,versicolor
5.6,2.7,4.2,1.3,versicolor
5.7,3.0,4.2,1.2,versicolor
5.7,2.9,4.2,1.3,versicolor
6.2,2.9,4.3,1.3,versicolor
5.1,2.5,3.0,1.1,versicolor
5.7,2.8,4.1,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
6.5,3.2,5.1,2.0,virginica
6.4,2.7,5.3,1.9,virginica
6.8,3.0,5.5,2.1,virginica
5.7,2.5,5.0,2.0,virginica
5.8,2.8,5.1,2.4,virginica
6.4,3.2,5.3,2.3,virginica
6.5,3.0,5.5,1.8,virginica
7.7,3.8,6.7,2.2,virginica
7.7,2.6,6.9,2.3,virginica
6.0,2.2,5.0,1.5,virginica
6.9,3.2,5.7,2.3,virginica
5.6,2.8,4.9,2.0,virginica
7.7,2.8,6.7,2.0,virginica
6.3,2.7,4.9,1.8,virginica
6.7,3.3,5.7,2.1,virginica
7.2,3.2,6.0,1.8,virginica
6.2,2.8,4.8,1.8,virginica
6.1,3.0,4.9,1.8,virginica
6.4,2.8,5.6,2.1,virginica
7.2,3.0,5.8,1.6,virginica
7.4,2.8,6.1,1.9,virginica
7.9,3.8,6.4,2.0,virginica
6.4,2.8,5.6,2.2,virginica
6.3,2.8,5.1,1.5,virginica
6.1,2.6,5.6,1.4,virginica
7.7,3.0,6.1,2.3,virginica
6.3,3.4,5.6,2.4,virginica
6.4,3.1,5.5,1.8,virginica
6.0,3.0,4.8,1.8,virginica
6.9,3.1,5.4,2.1,virginica
6.7,3.1,5.6,2.4,virginica
6.9,3.1,5.1,2.3,virginica
5.8,2.7,5.1,1.9,virginica
6.8,3.2,5.9,2.3,virginica
6.7,3.3,5.7,2.5,virginica
6.7,3.0,5.2,2.3,virginica
6.3,2.5,5.0,1.9,virginica
6.5,3.0,5.2,2.0,virginica
6.2,3.4,5.4,2.3,virginica
5.9,3.0,5.1,1.8,virginica"""

    # 运行实验
    results = classifier.run_experiment(iris_data)

    print(f"\n实验结果总结:")
    print(f"最优参数: {results['best_params']}")
    print(f"训练集准确率: {results['train_accuracy']:.4f}")
    print(f"测试集准确率: {results['test_accuracy']:.4f}")
    print(f"支持向量数量: {results['support_vectors']}")