import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def gaussian_kernel(size, sigma=1):
    """生成高斯核"""
    kernel = np.zeros((size, size))
    center = size // 2

    if sigma == 0:
        sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * sigma * sigma
    sum_val = 0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x * x + y * y) / s)
            sum_val += kernel[i, j]

    kernel /= sum_val
    return kernel


def sobel_filters(img):
    """应用Sobel算子计算梯度"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, theta):
    """非极大值抑制"""
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # 0度方向
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            # 45度方向
            elif 22.5 <= angle[i, j] < 67.5:
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            # 90度方向
            elif 67.5 <= angle[i, j] < 112.5:
                q = img[i + 1, j]
                r = img[i - 1, j]
            # 135度方向
            elif 112.5 <= angle[i, j] < 157.5:
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z


def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    """双阈值检测"""
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong


def edge_tracking(img, weak, strong=255):
    """边缘连接"""
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or
                        (img[i + 1, j + 1] == strong) or (img[i, j - 1] == strong) or
                        (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong) or
                        (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img


def canny_edge_detection_manual(img, sigma=1.4, kernel_size=5,
                                low_ratio=0.05, high_ratio=0.15):
    """手动实现完整的Canny边缘检测"""
    # 1. 高斯滤波
    gaussian = gaussian_kernel(kernel_size, sigma)
    img_smoothed = ndimage.convolve(img, gaussian)

    # 2. 计算梯度
    gradient_magnitude, gradient_direction = sobel_filters(img_smoothed)

    # 3. 非极大值抑制
    img_nms = non_max_suppression(gradient_magnitude, gradient_direction)

    # 4. 双阈值检测
    img_threshold, weak, strong = double_threshold(img_nms, low_ratio, high_ratio)

    # 5. 边缘连接
    img_final = edge_tracking(img_threshold, weak, strong)

    return img_final, img_smoothed, gradient_magnitude, img_nms


# ============ 主程序 ============

# 读取图像（使用你提供的新路径）
image_path = r'D:\Zeker\Documents\DigitalImageProcessing\Self-learning1Picture\building.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print(f"错误：无法读取图像文件")
    print(f"路径: {image_path}")
    print("\n请确认:")
    print("1. 文件确实存在于该路径")
    print("2. 文件名拼写正确")
    print("3. 文件没有被其他程序占用")
else:
    print("=" * 70)
    print("Canny边缘检测算法实验")
    print("=" * 70)
    print(f"✓ 成功读取图像")
    print(f"  图像路径: {image_path}")
    print(f"  图像尺寸: {image.shape[1]} × {image.shape[0]} 像素")
    print("=" * 70)

    # OpenCV内置Canny检测
    print("\n[1/6] 正在执行OpenCV内置Canny检测...")
    edges_opencv = cv2.Canny(image, 50, 150)
    print("     ✓ 完成")

    # 手动实现Canny检测
    print("\n[2/6] 正在执行手动实现的Canny检测...")
    edges_manual, img_smoothed, gradient_mag, img_nms = canny_edge_detection_manual(image)
    print("     ✓ 完成")

    # ============ 显示主要结果对比 ============
    print("\n[3/6] 生成主要结果对比图...")
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(edges_opencv, cmap='gray')
    plt.title('OpenCV Canny检测', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(edges_manual, cmap='gray')
    plt.title('手动实现Canny检测', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('canny_results.png', dpi=300, bbox_inches='tight')
    print("     ✓ 已保存: canny_results.png")
    plt.show()

    # ============ 显示各个处理步骤 ============
    print("\n[4/6] 生成处理步骤展示图...")
    plt.figure(figsize=(20, 4))

    plt.subplot(151)
    plt.imshow(image, cmap='gray')
    plt.title('1. 原始图像', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(img_smoothed, cmap='gray')
    plt.title('2. 高斯滤波\n(降噪处理)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(153)
    plt.imshow(gradient_mag, cmap='gray')
    plt.title('3. 梯度计算\n(Sobel算子)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(154)
    plt.imshow(img_nms, cmap='gray')
    plt.title('4. 非极大值抑制\n(边缘细化)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(155)
    plt.imshow(edges_manual, cmap='gray')
    plt.title('5. 双阈值与边缘连接\n(最终结果)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.suptitle('Canny边缘检测算法处理步骤', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('canny_steps.png', dpi=300, bbox_inches='tight')
    print("     ✓ 已保存: canny_steps.png")
    plt.show()

    # ============ 不同参数对比实验 ============
    print("\n[5/6] 进行参数对比实验...")

    # 不同sigma值对比
    print("     - 测试不同高斯滤波参数σ...")
    sigmas = [0.8, 1.4, 2.0]
    plt.figure(figsize=(15, 5))

    for idx, sigma in enumerate(sigmas):
        edges_temp, _, _, _ = canny_edge_detection_manual(image, sigma=sigma)
        plt.subplot(1, 3, idx + 1)
        plt.imshow(edges_temp, cmap='gray')
        plt.title(f'σ = {sigma}\n({"较弱平滑" if sigma < 1.2 else "中等平滑" if sigma < 1.8 else "较强平滑"})',
                  fontsize=14, fontweight='bold')
        plt.axis('off')

    plt.suptitle('不同高斯滤波参数σ对边缘检测的影响', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('canny_sigma_comparison.png', dpi=300, bbox_inches='tight')
    print("     ✓ 已保存: canny_sigma_comparison.png")
    plt.show()

    # 不同阈值对比
    print("     - 测试不同双阈值参数...")
    thresholds = [(0.05, 0.10), (0.05, 0.15), (0.05, 0.20)]
    plt.figure(figsize=(15, 5))

    for idx, (low, high) in enumerate(thresholds):
        edges_temp, _, _, _ = canny_edge_detection_manual(
            image, low_ratio=low, high_ratio=high
        )
        plt.subplot(1, 3, idx + 1)
        plt.imshow(edges_temp, cmap='gray')
        plt.title(f'高阈值比例 = {high}\n低阈值比例 = {low}', fontsize=12, fontweight='bold')
        plt.axis('off')

    plt.suptitle('不同双阈值参数对边缘检测的影响', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('canny_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("     ✓ 已保存: canny_threshold_comparison.png")
    plt.show()

    # ============ 统计信息 ============
    print("\n[6/6] 计算统计信息...")
    print("\n" + "=" * 70)
    print("实验统计结果")
    print("=" * 70)
    print(f"原始图像尺寸:              {image.shape[1]} × {image.shape[0]} 像素")
    print(f"总像素数:                  {image.size:,} 个")
    print("-" * 70)
    print(f"OpenCV检测边缘像素数:      {np.count_nonzero(edges_opencv):,} 个")
    print(f"OpenCV边缘像素占比:        {np.count_nonzero(edges_opencv) / image.size * 100:.2f}%")
    print("-" * 70)
    print(f"手动实现检测边缘像素数:    {np.count_nonzero(edges_manual):,} 个")
    print(f"手动实现边缘像素占比:      {np.count_nonzero(edges_manual) / image.size * 100:.2f}%")
    print("-" * 70)
    print(f"两种方法差异:              {abs(np.count_nonzero(edges_opencv) - np.count_nonzero(edges_manual)):,} 个像素")
    print(
        f"相似度:                    {(1 - abs(np.count_nonzero(edges_opencv) - np.count_nonzero(edges_manual)) / image.size) * 100:.2f}%")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("✓✓✓ 实验完成！所有结果图像已保存到当前目录 ✓✓✓")
    print("=" * 70)
    print("\n生成的文件列表:")
    print("  📊 1. canny_results.png           - 主要结果对比图")
    print("  📊 2. canny_steps.png             - 处理步骤展示图")
    print("  📊 3. canny_sigma_comparison.png  - sigma参数影响对比")
    print("  📊 4. canny_threshold_comparison.png - 阈值参数影响对比")
    print("\n💡 提示: 这些图片可以直接用于实验报告！")
    print("=" * 70)