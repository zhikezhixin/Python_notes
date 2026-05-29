import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class GLCMTextureExtractor:
    """灰度共生矩阵纹理特征提取器"""

    def __init__(self, image_path, distances=[1], angles=[0, 45, 90, 135], levels=256):
        """
        初始化GLCM特征提取器
        :param image_path: 图像路径
        :param distances: 距离列表
        :param angles: 角度列表(度)
        :param levels: 灰度级数
        """
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        self.distances = distances
        self.angles = [np.deg2rad(a) for a in angles]
        self.levels = levels

        # 量化灰度级
        self.quantized_image = self._quantize_image()

    def _quantize_image(self):
        """量化图像灰度级"""
        return (self.image * (self.levels - 1) / 255).astype(np.uint8)

    def compute_glcm(self, distance, angle):
        """
        计算灰度共生矩阵
        :param distance: 像素间距离
        :param angle: 角度(弧度)
        :return: 归一化的GLCM矩阵
        """
        rows, cols = self.quantized_image.shape
        glcm = np.zeros((self.levels, self.levels), dtype=np.float64)

        # 计算像素偏移量
        dx = int(np.round(distance * np.cos(angle)))
        dy = int(np.round(distance * np.sin(angle)))

        # 遍历图像构建共生矩阵
        for i in range(rows):
            for j in range(cols):
                # 计算邻居像素位置
                ni, nj = i + dy, j + dx

                # 检查邻居是否在图像范围内
                if 0 <= ni < rows and 0 <= nj < cols:
                    gray_i = self.quantized_image[i, j]
                    gray_j = self.quantized_image[ni, nj]
                    glcm[gray_i, gray_j] += 1

        # 归一化
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()

        return glcm

    def compute_all_statistics(self):
        """计算所有14种统计特征"""
        features = {}

        for dist in self.distances:
            for angle_deg, angle_rad in zip([0, 45, 90, 135], self.angles):
                glcm = self.compute_glcm(dist, angle_rad)

                key = f"d{dist}_a{angle_deg}"
                features[key] = {
                    '能量(Angular Second Moment)': self._angular_second_moment(glcm),
                    '对比度(Contrast)': self._contrast(glcm),
                    '相关性(Correlation)': self._correlation(glcm),
                    '方差(Variance)': self._variance(glcm),
                    '逆差矩(Inverse Difference Moment)': self._inverse_difference_moment(glcm),
                    '和平均(Sum Average)': self._sum_average(glcm),
                    '和方差(Sum Variance)': self._sum_variance(glcm),
                    '和熵(Sum Entropy)': self._sum_entropy(glcm),
                    '熵(Entropy)': self._entropy(glcm),
                    '差方差(Difference Variance)': self._difference_variance(glcm),
                    '差熵(Difference Entropy)': self._difference_entropy(glcm),
                    '信息测度相关1(Information Measure of Correlation 1)': self._info_measure_corr1(glcm),
                    '信息测度相关2(Information Measure of Correlation 2)': self._info_measure_corr2(glcm),
                    '最大相关系数(Maximal Correlation Coefficient)': self._maximal_correlation_coefficient(glcm)
                }

        return features

    def _angular_second_moment(self, glcm):
        """能量/角二阶矩"""
        return np.sum(glcm ** 2)

    def _contrast(self, glcm):
        """对比度"""
        i, j = np.meshgrid(range(self.levels), range(self.levels), indexing='ij')
        return np.sum(((i - j) ** 2) * glcm)

    def _correlation(self, glcm):
        """相关性"""
        i, j = np.meshgrid(range(self.levels), range(self.levels), indexing='ij')

        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)

        sigma_i = np.sqrt(np.sum(((i - mu_i) ** 2) * glcm))
        sigma_j = np.sqrt(np.sum(((j - mu_j) ** 2) * glcm))

        if sigma_i == 0 or sigma_j == 0:
            return 0

        return np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j)

    def _variance(self, glcm):
        """方差"""
        i, j = np.meshgrid(range(self.levels), range(self.levels), indexing='ij')
        mu = np.sum(i * glcm)
        return np.sum(((i - mu) ** 2) * glcm)

    def _inverse_difference_moment(self, glcm):
        """逆差矩/同质性"""
        i, j = np.meshgrid(range(self.levels), range(self.levels), indexing='ij')
        return np.sum(glcm / (1 + (i - j) ** 2))

    def _sum_average(self, glcm):
        """和平均"""
        p_x_plus_y = self._get_p_x_plus_y(glcm)
        k = np.arange(2 * self.levels)
        return np.sum(k * p_x_plus_y)

    def _sum_variance(self, glcm):
        """和方差"""
        p_x_plus_y = self._get_p_x_plus_y(glcm)
        sum_avg = self._sum_average(glcm)
        k = np.arange(2 * self.levels)
        return np.sum(((k - sum_avg) ** 2) * p_x_plus_y)

    def _sum_entropy(self, glcm):
        """和熵"""
        p_x_plus_y = self._get_p_x_plus_y(glcm)
        p_x_plus_y = p_x_plus_y[p_x_plus_y > 0]
        return -np.sum(p_x_plus_y * np.log2(p_x_plus_y))

    def _entropy(self, glcm):
        """熵"""
        glcm_pos = glcm[glcm > 0]
        return -np.sum(glcm_pos * np.log2(glcm_pos))

    def _difference_variance(self, glcm):
        """差方差"""
        p_x_minus_y = self._get_p_x_minus_y(glcm)
        k = np.arange(self.levels)
        mu = np.sum(k * p_x_minus_y)
        return np.sum(((k - mu) ** 2) * p_x_minus_y)

    def _difference_entropy(self, glcm):
        """差熵"""
        p_x_minus_y = self._get_p_x_minus_y(glcm)
        p_x_minus_y = p_x_minus_y[p_x_minus_y > 0]
        return -np.sum(p_x_minus_y * np.log2(p_x_minus_y))

    def _info_measure_corr1(self, glcm):
        """信息测度相关1"""
        hxy = self._entropy(glcm)
        hx = self._marginal_entropy_x(glcm)
        hy = self._marginal_entropy_y(glcm)
        hxy1 = self._hxy1(glcm)

        if max(hx, hy) == 0:
            return 0
        return (hxy - hxy1) / max(hx, hy)

    def _info_measure_corr2(self, glcm):
        """信息测度相关2"""
        hxy = self._entropy(glcm)
        hxy2 = self._hxy2(glcm)

        if hxy2 <= hxy:
            return 0
        return np.sqrt(1 - np.exp(-2 * (hxy2 - hxy)))

    def _maximal_correlation_coefficient(self, glcm):
        """最大相关系数"""
        px = np.sum(glcm, axis=1)
        py = np.sum(glcm, axis=0)

        # 避免除零
        px = np.where(px > 0, px, 1e-10)
        py = np.where(py > 0, py, 1e-10)

        Q = glcm / np.outer(px, py)

        try:
            eigvals = np.linalg.eigvals(Q)
            eigvals = np.sort(np.abs(eigvals))
            if len(eigvals) >= 2:
                return np.sqrt(eigvals[-2])
        except:
            pass

        return 0

    def _get_p_x_plus_y(self, glcm):
        """计算p(x+y)"""
        p_x_plus_y = np.zeros(2 * self.levels)
        for k in range(2 * self.levels):
            for i in range(self.levels):
                j = k - i
                if 0 <= j < self.levels:
                    p_x_plus_y[k] += glcm[i, j]
        return p_x_plus_y

    def _get_p_x_minus_y(self, glcm):
        """计算p(x-y)"""
        p_x_minus_y = np.zeros(self.levels)
        for k in range(self.levels):
            for i in range(self.levels):
                for j in range(self.levels):
                    if abs(i - j) == k:
                        p_x_minus_y[k] += glcm[i, j]
        return p_x_minus_y

    def _marginal_entropy_x(self, glcm):
        """边缘熵X"""
        px = np.sum(glcm, axis=1)
        px = px[px > 0]
        return -np.sum(px * np.log2(px))

    def _marginal_entropy_y(self, glcm):
        """边缘熵Y"""
        py = np.sum(glcm, axis=0)
        py = py[py > 0]
        return -np.sum(py * np.log2(py))

    def _hxy1(self, glcm):
        """HXY1"""
        px = np.sum(glcm, axis=1, keepdims=True)
        py = np.sum(glcm, axis=0, keepdims=True)
        pxpy = glcm[glcm > 0]
        px_expanded = np.repeat(px, self.levels, axis=1)[glcm > 0]
        py_expanded = np.repeat(py, self.levels, axis=0)[glcm > 0]

        log_term = px_expanded * py_expanded
        log_term = log_term[log_term > 0]
        pxpy = pxpy[:len(log_term)]

        return -np.sum(pxpy * np.log2(log_term))

    def _hxy2(self, glcm):
        """HXY2"""
        px = np.sum(glcm, axis=1, keepdims=True)
        py = np.sum(glcm, axis=0, keepdims=True)
        pxpy = px * py
        pxpy = pxpy[pxpy > 0]
        return -np.sum(pxpy * np.log2(pxpy))

    def visualize_results(self, features):
        """可视化结果"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('灰度共生矩阵纹理特征提取结果', fontsize=16, fontweight='bold')

        # 显示原始图像
        axes[0, 0].imshow(self.image, cmap='gray')
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')

        # 显示量化图像
        axes[0, 1].imshow(self.quantized_image, cmap='gray')
        axes[0, 1].set_title(f'量化图像({self.levels}级)', fontsize=12)
        axes[0, 1].axis('off')

        # 显示GLCM示例
        glcm_sample = self.compute_glcm(1, 0)
        im = axes[0, 2].imshow(glcm_sample, cmap='hot')
        axes[0, 2].set_title('GLCM示例(d=1, θ=0°)', fontsize=12)
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 绘制7个主要特征对比图
        feature_names = [
            '能量(Angular Second Moment)',
            '对比度(Contrast)',
            '相关性(Correlation)',
            '熵(Entropy)',
            '逆差矩(Inverse Difference Moment)',
            '方差(Variance)',
            '和熵(Sum Entropy)'
        ]

        # 从第二行开始绘制特征图
        plot_positions = [
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]

        for idx, feat_name in enumerate(feature_names[:6]):
            row, col = plot_positions[idx]

            values = [features[key][feat_name] for key in features.keys()]
            labels = list(features.keys())

            axes[row, col].bar(range(len(values)), values, color='steelblue', alpha=0.7)
            axes[row, col].set_title(feat_name, fontsize=10)
            axes[row, col].set_xticks(range(len(labels)))
            axes[row, col].set_xticklabels(labels, rotation=45, fontsize=8)
            axes[row, col].grid(True, alpha=0.3, linestyle='--')
            axes[row, col].set_ylabel('特征值', fontsize=9)

        plt.tight_layout()

        # 保存到与代码相同的目录
        save_path = 'glcm_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n结果图像已保存至: {save_path}")

        plt.show()

    def print_features(self, features):
        """打印特征值"""
        print("\n" + "=" * 80)
        print("灰度共生矩阵纹理特征提取结果")
        print("=" * 80)

        for key, feature_dict in features.items():
            print(f"\n{key} (距离={key[1]}, 角度={key.split('_')[1][1:]}°)")
            print("-" * 80)
            for feat_name, value in feature_dict.items():
                print(f"{feat_name:50s}: {value:12.6f}")


# 主程序
if __name__ == "__main__":
    # 图像路径
    image_path = r"D:\Zeker\Documents\DigitalImageProcessing\Self-learningPicture\building.png"

    try:
        # 创建特征提取器
        extractor = GLCMTextureExtractor(
            image_path=image_path,
            distances=[1],
            angles=[0, 45, 90, 135],
            levels=256
        )

        # 计算所有特征
        features = extractor.compute_all_statistics()

        # 打印结果
        extractor.print_features(features)

        # 可视化结果
        extractor.visualize_results(features)

        print("\n实验完成!")

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback

        traceback.print_exc()