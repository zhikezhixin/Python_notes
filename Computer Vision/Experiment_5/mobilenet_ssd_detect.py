"""
实验：基于 MobileNet_SSD + Caffe 的目标检测
支持图片检测 + 视频流检测
环境：opencv-contrib-python 4.13.0.92
"""

import cv2
import numpy as np
import time
import os

# ══════════════════════════════════════════════════════
# 路径配置（根据实际路径修改）
# ══════════════════════════════════════════════════════
BASE_DIR    = r"D:\CodeManager\PycharmProejcts\Python_notes\Computer Vision\Experiment_5"
MODEL_DIR   = os.path.join(BASE_DIR, "models")

IMAGE_PATH1 = os.path.join(BASE_DIR, "R-C.jpg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "R-C (1).jpg")
VIDEO_PATH  = os.path.join(BASE_DIR, "5月22日.mp4")

PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH    = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# ══════════════════════════════════════════════════════
# MobileNet_SSD 支持的 21 类（PASCAL VOC）
# ══════════════════════════════════════════════════════
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 为每个类别随机分配固定颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# 置信度阈值：低于此值的检测框将被忽略
CONFIDENCE_THRESHOLD = 0.5

# ══════════════════════════════════════════════════════
# 加载 Caffe 预训练模型
# cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
#   prototxt   ：网络结构定义文件
#   caffemodel ：包含预训练权重的模型文件
# ══════════════════════════════════════════════════════
print("[INFO] 正在加载 MobileNet_SSD 模型...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
print("[INFO] 模型加载成功！")


# ──────────────────────────────────────────────────────
# 核心函数：对单帧图像执行检测并返回标注后的图像
# ──────────────────────────────────────────────────────
def detect_on_frame(frame):
    """
    对一帧图像运行 MobileNet_SSD 目标检测

    参数:
        frame : BGR 格式的图像（numpy array）

    返回:
        output      : 绘制了检测框的图像
        results     : 检测结果列表 [(类别, 置信度, (x1,y1,x2,y2)), ...]
        infer_time  : 推理耗时（毫秒）
    """
    (h, w) = frame.shape[:2]

    # ── 步骤1：创建 Blob（块数据）──────────────────────
    # blobFromImage 将图像预处理为模型可接受的格式：
    #   resize → (300, 300)      MobileNet_SSD 固定输入尺寸
    #   scalefactor = 1/127.5    像素值从[0,255]缩放到[-1,1]
    #   mean = (127.5,127.5,127.5) 减去均值完成中心化
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5)
    )

    # ── 步骤2：设置输入并执行前向传播（推理）─────────────
    net.setInput(blob)
    t_start = time.time()
    detections = net.forward()          # 输出形状: (1, 1, N, 7)
    infer_time = (time.time() - t_start) * 1000  # 转为毫秒

    # ── 步骤3：解析并过滤检测结果 ───────────────────────
    # detections[0, 0, i] 包含 7 个值：
    # [图像ID, 类别ID, 置信度, x_min, y_min, x_max, y_max]
    # 其中坐标为归一化值（相对图像宽高的比例）
    results = []
    output  = frame.copy()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        # 过滤置信度低于阈值的结果
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(detections[0, 0, i, 1])
        label    = CLASSES[class_id]

        # 将归一化坐标 × 图像实际宽高，还原为像素坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # 边界修正，防止坐标越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        results.append((label, confidence, (x1, y1, x2, y2)))

        # ── 步骤4：绘制检测框和标签 ─────────────────────
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text    = f"{label}: {confidence:.2f}"
        y_text  = y1 - 10 if y1 - 10 > 10 else y1 + 15
        # 绘制标签背景矩形，提升文字可读性
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y_text - th - 4), (x1 + tw, y_text + 2), color, -1)
        cv2.putText(output, text, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 在左上角显示推理时间
    cv2.putText(output, f"SSD  {infer_time:.1f}ms",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return output, results, infer_time


# ══════════════════════════════════════════════════════
# 图片检测函数
# ══════════════════════════════════════════════════════
def detect_image(image_path):
    """对单张图片执行检测并显示结果"""
    # 使用 imdecode 读取（避免中文/特殊字符路径问题）
    raw = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return None, []

    output, results, infer_time = detect_on_frame(frame)

    print(f"\n[MobileNet_SSD] 图片检测结果 - {os.path.basename(image_path)}")
    print(f"  推理时间: {infer_time:.1f} ms | 检测目标数: {len(results)}")
    for (label, conf, box) in results:
        print(f"  ▸ {label:<15} 置信度={conf:.4f}  坐标={box}")

    # 保存结果
    save_name = f"ssd_result_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    save_path = os.path.join(BASE_DIR, save_name)
    cv2.imencode('.jpg', output)[1].tofile(save_path)
    print(f"  结果已保存: {save_path}")

    cv2.imshow("MobileNet_SSD - Image Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output, results


# ══════════════════════════════════════════════════════
# 视频检测函数
# ══════════════════════════════════════════════════════
def detect_video(video_path):
    """对视频流逐帧执行检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[MobileNet_SSD] 视频信息: {width}×{height}, {fps_orig:.1f}fps, 共{total}帧")

    # 配置视频写入器，保存检测结果视频
    save_path = os.path.join(BASE_DIR, "ssd_result_video.avi")
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    writer    = cv2.VideoWriter(save_path, fourcc, fps_orig, (width, height))

    frame_id   = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        frame_id += 1
        output, results, infer_time = detect_on_frame(frame)
        total_time += infer_time

        writer.write(output)  # 写入结果帧

        # 每10帧打印一次进度
        if frame_id % 10 == 0:
            print(f"  帧 {frame_id}/{total}  推理={infer_time:.1f}ms  检测={len(results)}个目标")

        cv2.imshow("MobileNet_SSD - Video Detection  [按Q退出]", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_time = total_time / frame_id if frame_id > 0 else 0
    print(f"\n[INFO] 视频检测完成 | 平均推理时间: {avg_time:.1f}ms | 结果已保存: {save_path}")
    return avg_time


# ══════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  MobileNet_SSD 目标检测")
    print("=" * 50)

    # 检测图片1
    print("\n[1/3] 检测图片: R-C.jpg")
    detect_image(IMAGE_PATH1)

    # 检测图片2
    print("\n[2/3] 检测图片: R-C (1).jpg")
    detect_image(IMAGE_PATH2)

    # 检测视频
    print("\n[3/3] 检测视频: 5月22日.mp4")
    detect_video(VIDEO_PATH)