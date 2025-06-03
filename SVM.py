import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体和图形参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def load_iris_data():
    """加载鸢尾花数据集"""
    # 模拟读取CSV数据（实际使用时替换为真实的文件读取）
    data = {
        'sepallength': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                        7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                        6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2],
        'sepalwidth': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                       3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7,
                       3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6],
        'petallength': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                        4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9,
                        6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1],
        'petalwidth': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                       1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4,
                       2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5],
        'class': ['setosa'] * 10 + ['versicolor'] * 10 + ['virginica'] * 10
    }

    # 创建完整的数据集（这里只是示例，实际应该读取完整的150条数据）
    # 为了演示，我们使用sklearn内置的iris数据集
    from sklearn.datasets import load_iris
    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    df['class'] = iris.target_names[iris.target]

    return df


def preprocess_data(df):
    """数据预处理"""
    # 特征和标签分离
    X = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values
    y = df['class'].values

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder, scaler


def train_svm_models(X_train, y_train):
    """训练不同核函数的SVM模型"""
    models = {
        'Linear SVM': SVC(kernel='linear', random_state=42),
        'RBF SVM': SVC(kernel='rbf', random_state=42),
        'Polynomial SVM': SVC(kernel='poly', degree=3, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} 训练完成")

    return trained_models


def evaluate_models(models, X_train, y_train, X_test, y_test, label_encoder):
    """评估模型性能"""
    results = {}

    for name, model in models.items():
        print(f"\n========== {name} 性能评估 ==========")

        # 训练集预测
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        # 测试集预测
        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_pred': test_pred
        }

        print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test, test_pred,
                                    target_names=label_encoder.classes_))

    return results


def plot_confusion_matrix(y_true, y_pred, labels, title):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{title} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(X, y, feature_names, class_names):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, feature in enumerate(feature_names):
        for j, class_name in enumerate(class_names):
            mask = y == j
            axes[i].hist(X[mask, i], alpha=0.7, label=class_name, bins=15)

        axes[i].set_title(f'{feature} 分布')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('频次')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    """模型性能对比图"""
    models = list(results.keys())
    train_accs = [results[model]['train_accuracy'] for model in models]
    test_accs = [results[model]['test_accuracy'] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_accs, width, label='训练集准确率', alpha=0.8)
    bars2 = ax.bar(x + width / 2, test_accs, width, label='测试集准确率', alpha=0.8)

    ax.set_xlabel('SVM模型')
    ax.set_ylabel('准确率')
    ax.set_title('不同SVM模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("========== SVM鸢尾花分类实验 ==========")

    # 1. 数据加载
    print("1. 加载数据...")
    df = load_iris_data()
    print(f"数据集形状: {df.shape}")
    print(f"类别分布:\n{df['class'].value_counts()}")

    # 2. 数据预处理
    print("\n2. 数据预处理...")
    X, y, label_encoder, scaler = preprocess_data(df)

    # 3. 数据划分
    print("\n3. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 4. 特征分布可视化
    print("\n4. 特征分布可视化...")
    feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    plot_feature_importance(X, y, feature_names, label_encoder.classes_)

    # 5. 训练SVM模型
    print("\n5. 训练SVM模型...")
    models = train_svm_models(X_train, y_train)

    # 6. 模型评估
    print("\n6. 模型评估...")
    results = evaluate_models(models, X_train, y_train, X_test, y_test, label_encoder)

    # 7. 结果可视化
    print("\n7. 结果可视化...")
    plot_model_comparison(results)

    # 8. 最佳模型的混淆矩阵
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_model = models[best_model_name]
    y_pred_best = results[best_model_name]['test_pred']

    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集准确率: {results[best_model_name]['test_accuracy']:.4f}")

    plot_confusion_matrix(y_test, y_pred_best, label_encoder.classes_,
                          f"最佳模型({best_model_name})")

    # 9. 实验总结
    print("\n========== 实验总结 ==========")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  训练集准确率: {result['train_accuracy']:.4f} ({result['train_accuracy'] * 100:.2f}%)")
        print(f"  测试集准确率: {result['test_accuracy']:.4f} ({result['test_accuracy'] * 100:.2f}%)")

    return models, results, label_encoder, scaler


if __name__ == "__main__":
    models, results, label_encoder, scaler = main()