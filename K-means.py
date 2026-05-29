import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans as SKLearnKMeans
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class KMeans:
    """K均值聚类算法实现"""

    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        # 聚类结果
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
        self.centroid_history = []

    def _init_centroids(self, X, method='random', custom_centroids=None):
        """初始化聚类中心"""
        if self.random_state:
            np.random.seed(self.random_state)

        if custom_centroids is not None:
            self.centroids = np.array(custom_centroids)
            return

        n_samples, n_features = X.shape

        if method == 'random':
            # 随机选择k个样本作为初始中心
            indices = np.random.choice(n_samples, self.k, replace=False)
            self.centroids = X[indices].copy()

        elif method == 'kmeans++':
            # K-means++初始化
            self.centroids = np.zeros((self.k, n_features))

            # 随机选择第一个中心
            self.centroids[0] = X[np.random.choice(n_samples)]

            for i in range(1, self.k):
                # 计算每个点到最近中心的距离
                distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in self.centroids[:i]]) for x in X])

                # 按概率选择下一个中心
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()

                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        self.centroids[i] = X[j]
                        break

        elif method == 'furthest':
            # 选择距离最远的点作为初始中心
            self.centroids = np.zeros((self.k, n_features))

            # 随机选择第一个中心
            self.centroids[0] = X[np.random.choice(n_samples)]

            for i in range(1, self.k):
                # 计算每个点到所有已选中心的最小距离
                distances = np.array([min([np.linalg.norm(x - c) for c in self.centroids[:i]]) for x in X])

                # 选择距离最远的点
                self.centroids[i] = X[np.argmax(distances)]

        elif method == 'corner':
            # 选择数据分布的角落点作为初始中心
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)

            if self.k == 2:
                self.centroids = np.array([
                    [min_vals[0], min_vals[1]],
                    [max_vals[0], max_vals[1]]
                ])
            elif self.k == 3:
                self.centroids = np.array([
                    [min_vals[0], min_vals[1]],
                    [max_vals[0], max_vals[1]],
                    [min_vals[0], max_vals[1]]
                ])
            elif self.k == 4:
                self.centroids = np.array([
                    [min_vals[0], min_vals[1]],
                    [max_vals[0], max_vals[1]],
                    [min_vals[0], max_vals[1]],
                    [max_vals[0], min_vals[1]]
                ])
            else:
                # 对于其他k值，使用随机初始化
                indices = np.random.choice(n_samples, self.k, replace=False)
                self.centroids = X[indices].copy()

    def _assign_clusters(self, X):
        """分配样本到最近的聚类中心"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """更新聚类中心"""
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def _calculate_inertia(self, X, labels):
        """计算簇内平方和"""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

    def fit(self, X, init_method='random', custom_centroids=None):
        """训练K均值模型"""
        # 初始化聚类中心
        self._init_centroids(X, init_method, custom_centroids)
        self.centroid_history = [self.centroids.copy()]

        for iteration in range(self.max_iters):
            # 分配样本到聚类
            labels = self._assign_clusters(X)

            # 更新聚类中心
            new_centroids = self._update_centroids(X, labels)

            # 检查收敛
            if np.allclose(self.centroids, new_centroids, rtol=self.tol):
                break

            self.centroids = new_centroids
            self.centroid_history.append(self.centroids.copy())

        self.labels = labels
        self.n_iter = iteration + 1
        self.inertia = self._calculate_inertia(X, labels)

        return self

    def predict(self, X):
        """预测新样本的聚类标签"""
        return self._assign_clusters(X)


def load_watermelon_data():
    """加载西瓜数据集4.0"""
    data = {
        '编号': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30],
        '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657,
                 0.360, 0.593, 0.719, 0.359, 0.339, 0.282, 0.748, 0.714, 0.483, 0.478, 0.525, 0.751, 0.532, 0.473,
                 0.725, 0.446],
        '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198,
                   0.370, 0.042, 0.103, 0.188, 0.241, 0.257, 0.232, 0.346, 0.312, 0.437, 0.369, 0.489, 0.472, 0.376,
                   0.445, 0.459]
    }

    df = pd.DataFrame(data)
    X = df[['密度', '含糖率']].values

    return X, df


def plot_clustering_results(X, results, title="K均值聚类结果对比"):
    """绘制聚类结果对比图"""
    n_experiments = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, (exp_name, kmeans_result) in enumerate(results.items()):
        if i >= 6:  # 最多显示6个子图
            break

        ax = axes[i]
        kmeans = kmeans_result['model']

        # 绘制数据点
        for cluster_id in range(kmeans.k):
            cluster_mask = kmeans.labels == cluster_id
            ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                       c=colors[cluster_id % len(colors)],
                       marker=markers[cluster_id % len(markers)],
                       s=100, alpha=0.7,
                       label=f'簇 {cluster_id + 1}')

        # 绘制聚类中心
        ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                   c='black', marker='x', s=200, linewidths=3, label='聚类中心')

        # 绘制聚类中心的演化轨迹
        for j in range(kmeans.k):
            centroid_path = np.array([hist[j] for hist in kmeans.centroid_history])
            ax.plot(centroid_path[:, 0], centroid_path[:, 1],
                    'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel('密度')
        ax.set_ylabel('含糖率')
        ax.set_title(f'{exp_name}\n(k={kmeans.k}, 迭代次数={kmeans.n_iter}, 惯性={kmeans.inertia:.3f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(results), 6):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def evaluate_clustering(X, labels, centroids, k):
    """评估聚类结果"""
    metrics = {}

    # 簇内平方和 (WCSS)
    wcss = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
    metrics['WCSS'] = wcss

    # 轮廓系数
    if k > 1 and len(np.unique(labels)) > 1:
        try:
            metrics['Silhouette'] = silhouette_score(X, labels)
        except:
            metrics['Silhouette'] = -1
    else:
        metrics['Silhouette'] = -1

    # 簇间距离
    if k > 1:
        inter_cluster_distances = []
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                inter_cluster_distances.append(dist)
        metrics['Min_Inter_Cluster_Distance'] = min(inter_cluster_distances)
        metrics['Avg_Inter_Cluster_Distance'] = np.mean(inter_cluster_distances)
    else:
        metrics['Min_Inter_Cluster_Distance'] = 0
        metrics['Avg_Inter_Cluster_Distance'] = 0

    return metrics


def run_kmeans_experiments():
    """运行K均值实验"""
    print("=== K均值算法在西瓜数据集4.0上的应用 ===\n")

    # 加载数据
    X, df = load_watermelon_data()
    print(f"数据集大小: {X.shape}")
    print(f"特征范围: 密度[{X[:, 0].min():.3f}, {X[:, 0].max():.3f}], 含糖率[{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    print()

    # 定义实验参数
    k_values = [2, 3, 4]  # 三组不同的k值

    # 三组不同的初始化方法
    init_methods = {
        '随机初始化': 'random',
        'K-means++': 'kmeans++',
        '边界点初始化': 'corner'
    }

    # 自定义初始中心点
    custom_centroids = {
        2: [[0.2, 0.1], [0.8, 0.5]],
        3: [[0.2, 0.1], [0.8, 0.5], [0.5, 0.3]],
        4: [[0.2, 0.1], [0.8, 0.5], [0.3, 0.4], [0.7, 0.2]]
    }

    # 存储所有实验结果
    all_results = {}
    evaluation_results = []

    print("开始实验...")
    print("=" * 80)

    # weather_scraper.py: 不同k值的比较 (使用K-means++初始化)
    print("weather_scraper.py: 不同k值的比较")
    print("-" * 40)

    for k in k_values:
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X, init_method='kmeans++')

        exp_name = f"k={k} (K-means++)"
        all_results[exp_name] = {'model': kmeans, 'type': 'k_comparison'}

        # 评估结果
        metrics = evaluate_clustering(X, kmeans.labels, kmeans.centroids, k)
        evaluation_results.append({
            'experiment': exp_name,
            'k': k,
            'init_method': 'K-means++',
            'iterations': kmeans.n_iter,
            'inertia': kmeans.inertia,
            **metrics
        })

        print(f"k={k}: 迭代{kmeans.n_iter}次, 惯性={kmeans.inertia:.4f}, 轮廓系数={metrics['Silhouette']:.4f}")

    print()

    # 实验2: 不同初始化方法的比较 (使用k=3)
    print("实验2: 不同初始化方法的比较 (k=3)")
    print("-" * 40)

    for init_name, init_method in init_methods.items():
        kmeans = KMeans(k=3, random_state=42)
        kmeans.fit(X, init_method=init_method)

        exp_name = f"{init_name} (k=3)"
        all_results[exp_name] = {'model': kmeans, 'type': 'init_comparison'}

        # 评估结果
        metrics = evaluate_clustering(X, kmeans.labels, kmeans.centroids, 3)
        evaluation_results.append({
            'experiment': exp_name,
            'k': 3,
            'init_method': init_name,
            'iterations': kmeans.n_iter,
            'inertia': kmeans.inertia,
            **metrics
        })

        print(f"{init_name}: 迭代{kmeans.n_iter}次, 惯性={kmeans.inertia:.4f}, 轮廓系数={metrics['Silhouette']:.4f}")

    print()

    # 实验3: 自定义初始中心点
    print("实验3: 自定义初始中心点")
    print("-" * 40)

    for k in [2, 3]:
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X, custom_centroids=custom_centroids[k])

        exp_name = f"自定义中心 (k={k})"
        all_results[exp_name] = {'model': kmeans, 'type': 'custom_init'}

        # 评估结果
        metrics = evaluate_clustering(X, kmeans.labels, kmeans.centroids, k)
        evaluation_results.append({
            'experiment': exp_name,
            'k': k,
            'init_method': '自定义',
            'iterations': kmeans.n_iter,
            'inertia': kmeans.inertia,
            **metrics
        })

        print(f"k={k}: 迭代{kmeans.n_iter}次, 惯性={kmeans.inertia:.4f}, 轮廓系数={metrics['Silhouette']:.4f}")
        print(f"  初始中心: {custom_centroids[k]}")

    print("\n" + "=" * 80)

    # 绘制结果
    plot_clustering_results(X, all_results)

    # 详细评估结果
    print("\n详细评估结果:")
    print("-" * 100)
    results_df = pd.DataFrame(evaluation_results)
    print(results_df.round(4))

    # 分析和建议
    print("\n" + "=" * 80)
    print("实验结果分析:")
    print("=" * 80)

    # 1. k值选择分析
    print("\n1. k值选择分析:")
    k_comparison = results_df[results_df['init_method'] == 'K-means++']
    best_k_silhouette = k_comparison.loc[k_comparison['Silhouette'].idxmax()]
    print(f"   - 最佳轮廓系数: k={best_k_silhouette['k']} (轮廓系数={best_k_silhouette['Silhouette']:.4f})")

    print("   - 惯性随k值变化:")
    for _, row in k_comparison.iterrows():
        print(f"     k={row['k']}: 惯性={row['inertia']:.4f}")

    # 2. 初始化方法分析
    print("\n2. 初始化方法分析 (k=3):")
    init_comparison = results_df[results_df['k'] == 3]
    best_init = init_comparison.loc[init_comparison['Silhouette'].idxmax()]
    print(f"   - 最佳初始化方法: {best_init['init_method']} (轮廓系数={best_init['Silhouette']:.4f})")

    print("   - 各方法比较:")
    for _, row in init_comparison.iterrows():
        print(
            f"     {row['init_method']}: 迭代{row['iterations']}次, 惯性={row['inertia']:.4f}, 轮廓系数={row['Silhouette']:.4f}")

    # 3. 初始中心选择建议
    print("\n3. 初始中心选择建议:")
    print("   有利于取得好结果的初始中心特点:")
    print("   - 应该分散在数据空间中，避免聚集在一起")
    print("   - K-means++方法通过概率选择远离已选中心的点，通常效果最好")
    print("   - 边界点初始化适合数据分布相对规整的情况")
    print("   - 随机初始化简单但结果不稳定，建议多次运行取最佳结果")
    print("   - 自定义中心点需要对数据分布有先验知识")

    # 4. 最佳配置推荐
    print("\n4. 最佳配置推荐:")
    best_overall = results_df.loc[results_df['Silhouette'].idxmax()]
    print(f"   推荐配置: k={best_overall['k']}, 初始化方法={best_overall['init_method']}")
    print(f"   性能指标: 轮廓系数={best_overall['Silhouette']:.4f}, 惯性={best_overall['inertia']:.4f}")

    return all_results, results_df


def plot_elbow_method(X):
    """绘制肘部法则图"""
    k_range = range(1, 8)
    inertias = []
    silhouettes = []

    for k in k_range:
        if k == 1:
            # k=1时的惯性就是所有点到质心的距离平方和
            centroid = X.mean(axis=0)
            inertia = np.sum((X - centroid) ** 2)
            inertias.append(inertia)
            silhouettes.append(0)  # k=1时轮廓系数无意义
        else:
            kmeans = KMeans(k=k, random_state=42)
            kmeans.fit(X, init_method='kmeans++')
            inertias.append(kmeans.inertia)

            if len(np.unique(kmeans.labels)) > 1:
                sil_score = silhouette_score(X, kmeans.labels)
                silhouettes.append(sil_score)
            else:
                silhouettes.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 肘部法则图
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('聚类数 k')
    ax1.set_ylabel('簇内平方和 (WCSS)')
    ax1.set_title('肘部法则图')
    ax1.grid(True, alpha=0.3)

    # 轮廓系数图
    ax2.plot(k_range[1:], silhouettes[1:], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('聚类数 k')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数图')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return k_range, inertias, silhouettes


def main():
    """主函数"""
    # 运行主要实验
    results, results_df = run_kmeans_experiments()

    # 绘制肘部法则图
    print("\n绘制肘部法则图和轮廓系数图...")
    X, _ = load_watermelon_data()
    plot_elbow_method(X)

    print("\n实验完成！")
    print("请查看上述图表和分析结果。")


if __name__ == "__main__":
    main()