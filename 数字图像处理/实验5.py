import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取图像
img_path = r"D:\Zeker\Documents\DigitalImageProcessing\Experiment 6 picture\Nine_graphics.png"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 形态学操作 - 闭运算填充边缘间隙
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# 查找轮廓
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建结果图像
result_img = img_rgb.copy()


# 定义形状识别函数
def get_shape_name(approx, vertices, aspect_ratio, circularity, solidity):
    """根据特征参数判断形状"""

    # 圆形: 高圆度
    if circularity > 0.85:
        return "圆形"

    # 三角形: 3个顶点
    if vertices == 3:
        return "三角形"

    # 四边形类
    if vertices == 4:
        # 通过长宽比区分正方形、长方形、菱形
        if 0.9 <= aspect_ratio <= 1.1:
            # 正方形或菱形 - 通过旋转角度区分
            rect = cv2.minAreaRect(approx)
            angle = rect[2]
            if abs(angle) > 35 and abs(angle) < 55:
                return "菱形"
            else:
                return "正方形"
        else:
            return "长方形"

    # 五边形: 5个顶点
    if vertices == 5:
        return "五边形"

    # 五角星: 10个顶点(凹多边形)
    if vertices == 10 and solidity < 0.7:
        return "五角星"

    # 十字形: 12个顶点且实度较低
    if vertices >= 12 and solidity < 0.75:
        return "十字形"

    # 四叶花形: 多个顶点且圆度较低
    if vertices >= 8 and circularity < 0.7 and solidity < 0.85:
        return "四叶花形"

    return f"未知({vertices}边形)"


def get_color_name(hsv_color):
    """根据HSV值判断颜色"""
    h, s, v = hsv_color

    # 颜色判断规则
    if s < 50:  # 低饱和度
        if v > 200:
            return "白色"
        elif v < 50:
            return "黑色"
        else:
            return "灰色"

    # 根据色调判断颜色
    if h < 10 or h > 170:
        return "红色"
    elif 10 <= h < 25:
        return "橙色"
    elif 25 <= h < 35:
        return "黄色"
    elif 35 <= h < 85:
        return "绿色"
    elif 85 <= h < 125:
        return "蓝色"
    elif 125 <= h < 155:
        return "紫色"
    elif 155 <= h < 170:
        return "粉色"

    return "未知颜色"


# 存储检测结果
shapes_info = []

# 遍历每个轮廓
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    # 过滤小轮廓
    if area < 1000:
        continue

    # 轮廓近似
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 计算特征
    vertices = len(approx)

    # 外接矩形
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0

    # 圆度
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0

    # 实度(solidity)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0

    # 形状识别
    shape_name = get_shape_name(approx, vertices, aspect_ratio, circularity, solidity)

    # 颜色识别 - 提取ROI区域的主要颜色
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    mean_color = cv2.mean(img, mask=mask)[:3]

    # 转换为HSV进行颜色判断
    bgr_color = np.uint8([[mean_color]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    color_name = get_color_name(hsv_color)

    # 存储信息
    shapes_info.append({
        'id': len(shapes_info) + 1,
        'shape': shape_name,
        'color': color_name,
        'vertices': vertices,
        'area': area,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'position': (x, y)
    })

    # 在图像上标注
    cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(result_img, [approx], -1, (255, 0, 0), 3)

    # 计算文本位置
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = x + w // 2, y + h // 2

    # 添加标签
    label = f"{len(shapes_info)}: {color_name}{shape_name}"
    cv2.putText(result_img, str(len(shapes_info)), (cX - 10, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# 按位置排序(从上到下,从左到右)
shapes_info.sort(key=lambda x: (x['position'][1] // 200, x['position'][0]))

# 重新编号
for i, shape in enumerate(shapes_info):
    shape['id'] = i + 1

# 创建可视化结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('图像形状检测处理流程', fontsize=16, fontweight='bold')

# 原始图像
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('(a) 原始图像')
axes[0, 0].axis('off')

# 灰度图
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('(b) 灰度图像')
axes[0, 1].axis('off')

# 边缘检测
axes[0, 2].imshow(edges, cmap='gray')
axes[0, 2].set_title('(c) Canny边缘检测')
axes[0, 2].axis('off')

# 形态学处理
axes[1, 0].imshow(closed, cmap='gray')
axes[1, 0].set_title('(d) 形态学闭运算')
axes[1, 0].axis('off')

# 轮廓检测
contour_img = img_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
axes[1, 1].imshow(contour_img)
axes[1, 1].set_title('(e) 轮廓检测结果')
axes[1, 1].axis('off')

# 最终识别结果
axes[1, 2].imshow(result_img)
axes[1, 2].set_title('(f) 形状识别与标注')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('shape_detection_process.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印检测结果
print("=" * 80)
print("图像形状检测识别结果")
print("=" * 80)
print(f"{'编号':<6} {'形状':<12} {'颜色':<10} {'顶点数':<8} {'面积':<10} {'圆度':<8} {'长宽比':<8} {'实度':<8}")
print("-" * 80)

for shape in shapes_info:
    print(f"{shape['id']:<6} {shape['shape']:<12} {shape['color']:<10} "
          f"{shape['vertices']:<8} {shape['area']:<10.0f} "
          f"{shape['circularity']:<8.3f} {shape['aspect_ratio']:<8.3f} "
          f"{shape['solidity']:<8.3f}")

print("=" * 80)

# 创建详细特征对比图
fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# 准备表格数据
table_data = [['编号', '识别结果', '颜色', '顶点数', '圆度', '长宽比', '实度', '面积']]
for shape in shapes_info:
    table_data.append([
        str(shape['id']),
        shape['shape'],
        shape['color'],
        str(shape['vertices']),
        f"{shape['circularity']:.3f}",
        f"{shape['aspect_ratio']:.3f}",
        f"{shape['solidity']:.3f}",
        f"{shape['area']:.0f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.08, 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# 设置表头样式
for i in range(8):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置数据行颜色
for i in range(1, len(table_data)):
    for j in range(8):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('形状检测特征参数对比表', fontsize=14, fontweight='bold', pad=20)
plt.savefig('shape_features_table.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n处理完成!图像已保存。")