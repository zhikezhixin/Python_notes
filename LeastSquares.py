import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LinearLeastSquares:
    """
    最小二乘法线性回归器
    """

    def __init__(self):
        self.weights = None
        self.bias = None
        self.fitted = False

    def fit(self, X, y):
        """
        训练最小二乘法模型

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
        """
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)

        # 确保X是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 添加偏置项（截距项）
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # 最小二乘法求解: theta = (X^T * X)^(-1) * X^T * y
        try:
            # 计算 X^T * X
            XTX = np.dot(X_with_bias.T, X_with_bias)
            # 计算 X^T * y
            XTy = np.dot(X_with_bias.T, y)
            # 求解参数
            theta = np.linalg.solve(XTX, XTy)

            self.bias = theta[0]
            self.weights = theta[1:]
            self.fitted = True

            print("最小二乘法求解过程:")
            print(f"X^T * X 矩阵:\n{XTX}")
            print(f"X^T * y 向量: {XTy}")
            print(f"求解得到的参数 θ: {theta}")

        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
            self.fitted = True
            print("使用伪逆求解参数")

    def predict(self, X):
        """
        预测

        参数:
        X: 特征矩阵 (n_samples, n_features)

        返回:
        预测结果 (n_samples,)
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 线性预测: y = X * w + b
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def get_equation(self):
        """
        返回拟合的线性方程
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")

        if len(self.weights) == 1:
            return f"y = {self.weights[0]:.4f} * x + {self.bias:.4f}"
        else:
            terms = [f"{w:.4f} * x{i + 1}" for i, w in enumerate(self.weights)]
            return f"y = {' + '.join(terms)} + {self.bias:.4f}"


def load_csv_data(file_path=None):
    """
    读取CSV数据文件

    参数:
    file_path: CSV文件路径，如果为None则提示用户输入
    """
    try:
        # 如果没有提供文件路径，提示用户输入
        if file_path is None:
            print("请输入CSV文件的完整路径（例如：C:/Users/YourName/Desktop/data.csv）")
            file_path = input("文件路径: ").strip()

            # 移除可能的引号
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
            elif file_path.startswith("'") and file_path.endswith("'"):
                file_path = file_path[1:-1]

        print(f"正在读取文件: {file_path}")

        # 尝试使用pandas读取CSV文件
        try:
            # 首先尝试有列名的情况
            data = pd.read_csv(file_path)
            print(f"文件列名: {list(data.columns)}")

            # 如果只有两列，直接使用
            if data.shape[1] == 2:
                X = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
            else:
                print(f"数据有{data.shape[1]}列，请选择要使用的列：")
                print("列索引：", list(range(data.shape[1])))
                print("列名：", list(data.columns))

                x_col = int(input("输入特征列的索引（从0开始）: "))
                y_col = int(input("输入目标列的索引（从0开始）: "))

                X = data.iloc[:, x_col].values
                y = data.iloc[:, y_col].values

        except Exception:
            # 如果pandas读取失败，尝试手动解析（无列名）
            print("使用pandas读取失败，尝试手动解析...")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    values = line.split(',')
                    if len(values) >= 2:
                        x_val = float(values[0])
                        y_val = float(values[1])
                        data.append([x_val, y_val])
                except ValueError as e:
                    print(f"第{i + 1}行数据格式错误，跳过: {line}")
                    continue

            if not data:
                raise ValueError("没有有效的数据行")

            data = np.array(data)
            X = data[:, 0]
            y = data[:, 1]

        print(f"成功加载数据: {len(X)} 个样本")
        print(f"X范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"y范围: [{y.min():.2f}, {y.max():.2f}]")
        print(f"前5行数据预览:")
        for i in range(min(5, len(X))):
            print(f"  X={X[i]:.2f}, y={y[i]:.2f}")

        return X, y

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        print("请检查文件路径是否正确")
        return None, None
    except Exception as e:
        print(f"读取数据失败: {e}")
        print("请确保：")
        print("1. 文件路径正确")
        print("2. 文件是CSV格式")
        print("3. 数据至少有两列（特征和目标值）")
        print("4. 数据是数值型的")
        return None, None


def calculate_metrics(y_true, y_pred):
    """
    计算回归评估指标
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 计算平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


def plot_results(X, y, model, X_test=None, y_test=None, y_pred=None):
    """决策树
    可视化回归结果
    """
    plt.figure(figsize=(15, 5))

    # 子图1：原始数据散点图
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.7, color='blue', label='原始数据')
    plt.title("原始数据分布")
    plt.xlabel("X (特征)")
    plt.ylabel("y (目标值)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2：拟合结果
    plt.subplot(1, 3, 2)
    plt.scatter(X, y, alpha=0.7, color='blue', label='训练数据')

    # 绘制拟合直线
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'拟合直线')

    plt.title("最小二乘法拟合结果")
    plt.xlabel("X (特征)")
    plt.ylabel("y (目标值)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3：预测 vs 真实值
    plt.subplot(1, 3, 3)
    if X_test is not None and y_test is not None and y_pred is not None:
        plt.scatter(y_test, y_pred, alpha=0.7, color='green')

        # 绘制完美预测线 (y=x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title("预测值 vs 真实值")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # 如果没有测试数据，显示残差图
        y_pred_all = model.predict(X)
        residuals = y - y_pred_all
        plt.scatter(y_pred_all, residuals, alpha=0.7, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title("残差图")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_data(X, y):
    """
    数据分析
    """
    print("\n数据分析:")
    print(f"样本数量: {len(X)}")
    print(f"特征统计:")
    print(f"  均值: {X.mean():.4f}")
    print(f"  标准差: {X.std():.4f}")
    print(f"  最小值: {X.min():.4f}")
    print(f"  最大值: {X.max():.4f}")

    print(f"目标变量统计:")
    print(f"  均值: {y.mean():.4f}")
    print(f"  标准差: {y.std():.4f}")
    print(f"  最小值: {y.min():.4f}")
    print(f"  最大值: {y.max():.4f}")

    # 计算相关系数
    correlation = np.corrcoef(X, y)[0, 1]
    print(f"相关系数: {correlation:.4f}")


def main():
    """
    主函数：执行完整的实验流程
    """
    print("=" * 60)
    print("最小二乘法线性回归实验")
    print("=" * 60)

    # 1. 数据加载
    print("\n1. 数据加载...")
    X, y = load_csv_data()  # 会提示用户输入文件路径

    if X is None or y is None:
        print("数据加载失败！")
        return

    # 2. 数据分析
    analyze_data(X, y)

    # 3. 数据划分
    print("\n2. 数据划分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 4. 模型训练
    print("\n3. 模型训练...")
    model = LinearLeastSquares()
    model.fit(X_train, y_train)

    print(f"\n拟合的线性方程: {model.get_equation()}")

    # 5. 模型预测和评估
    print("\n4. 模型评估...")

    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 6. 结果可视化
    print("\n5. 结果可视化...")
    plot_results(X, y, model, X_test, y_test, y_test_pred)

    # 7. 算法原理说明
    print("\n6. 算法原理:")
    print("最小二乘法通过最小化残差平方和来求解最优参数：")
    print("目标函数: min Σ(yi - (w*xi + b))²")
    print("解析解: θ = (X^T * X)^(-1) * X^T * y")
    print("其中 X 是增广特征矩阵（包含偏置项），θ = [b, w]^T")

    # 8. 预测示例
    print("\n7. 预测示例:")
    test_points = [1, 10, 20, 35]
    for point in test_points:
        pred = model.predict([point])[0]
        print(f"当 x = {point} 时，预测 y = {pred:.4f}")

    return model, test_metrics['R²']


# 运行实验
if __name__ == "__main__":
    model, r2_score = main()

    print(f"\n实验完成！")
    print(f"模型决定系数 R² = {r2_score:.4f} ({r2_score * 100:.2f}%)")
    print("R²越接近1，模型拟合效果越好")