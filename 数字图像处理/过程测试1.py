# hist_specification.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号


def imread_chinese(image_path):
    """
    读取包含中文路径的图像文件
    解决OpenCV对中文路径支持不好的问题
    """
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")

    # 使用numpy读取图像数据
    img_array = np.fromfile(image_path, dtype=np.uint8)
    # 解码图像
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法解码图像: {image_path}")

    return img


def imwrite_chinese(image_path, img):
    """
    保存图像到包含中文的路径
    """
    # 编码图像
    ext = image_path.split('.')[-1]
    success, img_encode = cv2.imencode(f'.{ext}', img)

    if success:
        # 写入文件
        img_encode.tofile(image_path)
        return True
    return False


def hist_specification(image_path, pz):
    """
    直方图规定化实现

    参数:
        image_path: 输入图像路径
        pz: 目标概率分布（长度256的numpy数组）

    返回:
        包含所有结果数据的字典
    """
    # 读取图像（支持中文路径）
    I = imread_chinese(image_path)

    rows, cols = I.shape
    N = rows * cols
    print(f"图像尺寸: {rows} × {cols}, 总像素数: {N}")

    # 步骤1: 计算原始图像的直方图和PMF
    hr, _ = np.histogram(I.flatten(), bins=256, range=(0, 256))
    pr = hr.astype(np.float64) / N

    # 步骤2: 计算原始图像的CDF
    S = np.cumsum(pr)

    # 步骤3: 处理目标PMF
    pz = np.asarray(pz, dtype=np.float64).flatten()
    if pz.size != 256:
        raise ValueError("目标PMF必须是长度为256的数组")
    pz = pz / pz.sum()  # 归一化

    # 计算目标CDF
    G = np.cumsum(pz)

    # 步骤4: 构造映射表T[k]
    T = np.zeros(256, dtype=np.uint8)
    for k in range(256):
        s_val = S[k]
        # 找到最小的ℓ使得G[ℓ] >= S[k]
        ell = np.searchsorted(G, s_val, side='left')
        if ell >= 256:
            ell = 255
        T[k] = np.uint8(ell)

    print(f"映射表构造完成，示例: T[0]={T[0]}, T[128]={T[128]}, T[255]={T[255]}")

    # 步骤5: 应用映射（向量化操作）
    J = T[I]

    # 计算变换后图像的实际直方图
    hz, _ = np.histogram(J.flatten(), bins=256, range=(0, 256))
    pz_emp = hz.astype(np.float64) / N

    # 计算匹配误差
    error_L1 = np.sum(np.abs(pz - pz_emp))
    print(f"目标直方图与实际直方图的L1距离: {error_L1:.6f}")

    # 可视化结果
    plot_results(I, J, pr, pz, pz_emp, T)

    # 保存结果到与输入图像相同的目录
    import os
    output_dir = os.path.dirname(image_path)
    transformed_path = os.path.join(output_dir, 'transformed_image.png')
    results_path = os.path.join(output_dir, 'histogram_specification_results.npz')
    figure_path = os.path.join(output_dir, 'histogram_specification_results.png')

    # 保存变换后图像（支持中文路径）
    imwrite_chinese(transformed_path, J)

    # 保存数据
    np.savez(results_path,
             original=I, transformed=J, pr=pr, pz=pz,
             pz_emp=pz_emp, T=T, S=S, G=G)

    print(f"\n结果已保存:")
    print(f"  - 变换后图像: {transformed_path}")
    print(f"  - 数据文件: {results_path}")
    print(f"  - 可视化结果: {figure_path}")

    return {
        'original': I,
        'transformed': J,
        'pr': pr,
        'pz_desired': pz,
        'pz_empirical': pz_emp,
        'mapping_T': T,
        'S': S,
        'G': G,
        'error_L1': error_L1
    }


def plot_results(I, J, pr, pz, pz_emp, T):
    """绘制所有实验结果"""
    fig = plt.figure(figsize=(15, 8))

    # 第一行: 图像对比
    plt.subplot(2, 3, 1)
    plt.imshow(I, cmap='gray', vmin=0, vmax=255)
    plt.title('(a) 原始图像', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(J, cmap='gray', vmin=0, vmax=255)
    plt.title('(b) 变换后图像', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.plot(np.arange(256), T, 'b-', linewidth=2)
    plt.plot([0, 255], [0, 255], 'r--', alpha=0.5, label='y=x参考线')
    plt.xlabel('输入灰度级 r', fontsize=10)
    plt.ylabel('输出灰度级 z = T(r)', fontsize=10)
    plt.title('(c) 强度变换函数', fontsize=12, fontweight='bold')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 第二行: 直方图对比
    plt.subplot(2, 3, 4)
    plt.bar(np.arange(256), pr, width=1.0, color='blue', alpha=0.7)
    plt.xlabel('灰度级', fontsize=10)
    plt.ylabel('概率', fontsize=10)
    plt.title('(d) 原始图像直方图 p_r(r)', fontsize=12, fontweight='bold')
    plt.xlim([0, 255])
    y_max = max(pr) * 1.1 if max(pr) > 0 else 0.01
    plt.ylim([0, y_max])

    plt.subplot(2, 3, 5)
    plt.bar(np.arange(256), pz, width=1.0, color='green', alpha=0.7)
    plt.xlabel('灰度级', fontsize=10)
    plt.ylabel('概率', fontsize=10)
    plt.title('(e) 目标直方图 p_z(z) (期望)', fontsize=12, fontweight='bold')
    plt.xlim([0, 255])
    y_max = max(pz) * 1.1 if max(pz) > 0 else 0.01
    plt.ylim([0, y_max])

    plt.subplot(2, 3, 6)
    plt.bar(np.arange(256), pz_emp, width=1.0, color='red', alpha=0.7)
    plt.xlabel('灰度级', fontsize=10)
    plt.ylabel('概率', fontsize=10)
    plt.title('(f) 变换后直方图 p_z(z) (实际)', fontsize=12, fontweight='bold')
    plt.xlim([0, 255])
    y_max = max(pz_emp) * 1.1 if max(pz_emp) > 0 else 0.01
    plt.ylim([0, y_max])

    plt.tight_layout()

    # 保存图片（支持中文路径）
    plt.savefig('histogram_specification_results.png', dpi=300, bbox_inches='tight')
    print("\n可视化图表已生成")
    plt.show()


def create_target_histogram_from_fig1b():
    """
    根据Fig.1(b)创建目标直方图
    基于Part A的理论结果: z = r/2
    这意味着目标分布应在[0, 127]区间具有较高密度
    """
    pz = np.zeros(256)

    # 根据连续情形的结果，目标分布在[0, 0.5]即离散域[0, 127]有均匀分布
    # 对应PDF p_z(z) = 2 在 z ∈ [0, 0.5]
    pz[0:128] = 1.0  # 在前半区间均匀分布
    pz[128:256] = 0.0  # 后半区间概率为0

    # 归一化（确保总和为1）
    pz = pz / pz.sum()

    return pz


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("数字图像处理 - 直方图规定化实验")
    print("=" * 60)

    # 指定图像路径
    image_path = r"D:\Zeker\Documents\数字图像处理\过程考试1测试图片\猫的灰度图片.jpg"

    # 检查文件是否存在
    import os

    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        print("请检查文件路径是否正确！")
        exit(1)

    print(f"\n读取图像: {image_path}")

    # 创建目标直方图（根据Fig.1(b)和Part A的理论结果）
    print("\n创建目标直方图...")
    pz_target = create_target_histogram_from_fig1b()
    print(f"目标直方图: 在灰度级[0, 127]区间有非零值")
    print(f"目标直方图总和: {pz_target.sum():.6f} (应为1.0)")

    # 执行直方图规定化
    print("\n开始执行直方图规定化...")
    print("-" * 60)

    try:
        results = hist_specification(image_path, pz_target)

        # 输出统计信息
        print("\n" + "=" * 60)
        print("实验结果统计:")
        print("=" * 60)
        print(f"原始图像统计:")
        print(f"  - 平均亮度: {np.mean(results['original']):.2f}")
        print(f"  - 标准差: {np.std(results['original']):.2f}")
        print(f"  - 最小值: {np.min(results['original'])}")
        print(f"  - 最大值: {np.max(results['original'])}")

        print(f"\n变换后图像统计:")
        print(f"  - 平均亮度: {np.mean(results['transformed']):.2f}")
        print(f"  - 标准差: {np.std(results['transformed']):.2f}")
        print(f"  - 最小值: {np.min(results['transformed'])}")
        print(f"  - 最大值: {np.max(results['transformed'])}")

        brightness_change = np.mean(results['transformed']) - np.mean(results['original'])
        contrast_change = (np.std(results['transformed']) / np.std(results['original']) - 1) * 100

        print(f"\n变化分析:")
        print(
            f"  - 亮度变化: {brightness_change:.2f} (变化率: {brightness_change / np.mean(results['original']) * 100:.2f}%)")
        print(f"  - 对比度变化率: {contrast_change:.2f}%")

        print("\n" + "=" * 60)
        print("实验完成！所有结果已保存。")
        print("=" * 60)

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback

        traceback.print_exc()