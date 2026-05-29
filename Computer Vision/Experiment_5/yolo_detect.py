"""
实验：基于 YOLOv3 + Darknet 的目标检测
支持图片检测 + 视频流检测
环境：opencv-contrib-python 4.13.0.92
"""

import cv2
import numpy as np
import time
import os

# ══════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════
BASE_DIR    = r"D:\CodeManager\PycharmProejcts\Python_notes\Computer Vision\Experiment_5"
MODEL_DIR   = os.path.join(BASE_DIR, "models")

IMAGE_PATH1 = os.path.join(BASE_DIR, "R-C.jpg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "R-C (1).jpg")
VIDEO_PATH  = os.path.join(BASE_DIR, "5月22日.mp4")

CONFIG_PATH  = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
LABELS_PATH  = os.path.join(MODEL_DIR, "coco.names")

# ══════════════════════════════════════════════════════
# 检测参数
# ══════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.5   # 置信度阈值
NMS_THRESHOLD        = 0.4   # 非极大值抑制阈值（去除冗余重叠框）
INPUT_SIZE           = 416   # YOLOv3 标准输入尺寸

# ══════════════════════════════════════════════════════
# 加载类别标签（COCO 80类）
# ══════════════════════════════════════════════════════
with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]
print(f"[INFO] 加载类别标签完成，共 {len(LABELS)} 类")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# ══════════════════════════════════════════════════════
# 加载 Darknet 预训练模型
# cv2.dnn.readNetFromDarknet(cfg, weights)
#   cfg     ：YOLOv3 网络结构配置文件
#   weights ：预训练权重文件
# ══════════════════════════════════════════════════════
print("[INFO] 正在加载 YOLOv3 模型（文件较大，请稍候）...")
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
print("[INFO] 模型加载成功！")

# 获取 YOLOv3 的三个检测输出层名称
# YOLOv3 在三个不同尺度进行预测：
#   82层(13×13)检测大目标、94层(26×26)检测中目标、106层(52×52)检测小目标
all_layers    = net.getLayerNames()
output_layers = [all_layers[i - 1] for i in net.getUnconnectedOutLayers()]
print(f"[INFO] YOLOv3 输出层: {output_layers}")


# ──────────────────────────────────────────────────────
# 核心函数：对单帧图像执行 YOLO 检测
# ──────────────────────────────────────────────────────
def detect_on_frame(frame):
    """
    对一帧图像运行 YOLOv3 目标检测

    参数:
        frame : BGR 格式的图像（numpy array）

    返回:
        output     : 绘制了检测框的图像
        results    : 检测结果列表 [(类别, 置信度, (x1,y1,x2,y2)), ...]
        infer_time : 推理耗时（毫秒）
    """
    (h, w) = frame.shape[:2]

    # ── 步骤1：创建 Blob ────────────────────────────────
    # blobFromImage 参数说明：
    #   scalefactor = 1/255.0   像素值归一化到 [0, 1]
    #   size = (416, 416)       YOLOv3 标准输入尺寸
    #   mean = (0, 0, 0)        不减均值
    #   swapRB = True           OpenCV 默认 BGR，转换为 YOLO 所需的 RGB
    #   crop = False            直接缩放而不裁剪
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    # ── 步骤2：执行前向传播，同时获取三个尺度的输出 ────────
    net.setInput(blob)
    t_start = time.time()
    layer_outputs = net.forward(output_layers)  # 返回3个检测头的输出
    infer_time = (time.time() - t_start) * 1000

    # ── 步骤3：解析三个检测头的输出 ────────────────────────
    # 每个检测结果的格式（共 5 + 80 = 85 个值）：
    # [cx, cy, w, h, objectness, p_cls0, p_cls1, ..., p_cls79]
    # cx,cy,w,h 均为相对于图像宽高的归一化值
    boxes       = []
    confidences = []
    class_ids   = []

    for output in layer_outputs:           # 遍历3个检测头
        for detection in output:           # 遍历该尺度所有候选框
            scores    = detection[5:]                  # 80个类别的概率
            class_id  = int(np.argmax(scores))         # 取概率最高的类别
            confidence = float(scores[class_id])       # 该类别的置信度

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # 将归一化的中心点坐标和宽高还原为像素值
            cx = int(detection[0] * w)
            cy = int(detection[1] * h)
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)

            # 转换为左上角坐标（OpenCV 矩形格式）
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))

            boxes.append([x1, y1, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

    # ── 步骤4：非极大值抑制（NMS）────────────────────────
    # 同一目标可能被多个网格重复检测，NMS 保留置信度最高的框
    # NMS_THRESHOLD：IoU 超过此值的重叠框将被抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # ── 步骤5：绘制最终检测结果 ─────────────────────────
    results = []
    output_frame = frame.copy()

    if len(indices) > 0:
        for i in indices.flatten():
            (x1, y1, bw, bh) = boxes[i]
            x2 = min(w - 1, x1 + bw)
            y2 = min(h - 1, y1 + bh)
            confidence = confidences[i]
            class_id   = class_ids[i]
            label      = LABELS[class_id]

            results.append((label, confidence, (x1, y1, x2, y2)))

            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            text   = f"{label}: {confidence:.2f}"
            y_text = y1 - 10 if y1 - 10 > 10 else y1 + 15
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_frame,
                          (x1, y_text - th - 4), (x1 + tw, y_text + 2), color, -1)
            cv2.putText(output_frame, text, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 左上角显示推理时间
    cv2.putText(output_frame, f"YOLO  {infer_time:.1f}ms",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return output_frame, results, infer_time


# ══════════════════════════════════════════════════════
# 图片检测
# ══════════════════════════════════════════════════════
def detect_image(image_path):
    raw   = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return None, []

    output, results, infer_time = detect_on_frame(frame)

    print(f"\n[YOLOv3] 图片检测结果 - {os.path.basename(image_path)}")
    print(f"  推理时间: {infer_time:.1f} ms | 检测目标数: {len(results)}")
    for (label, conf, box) in results:
        print(f"  ▸ {label:<15} 置信度={conf:.4f}  坐标={box}")

    save_name = f"yolo_result_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    save_path = os.path.join(BASE_DIR, save_name)
    cv2.imencode('.jpg', output)[1].tofile(save_path)
    print(f"  结果已保存: {save_path}")

    cv2.imshow("YOLOv3 - Image Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output, results


# ══════════════════════════════════════════════════════
# 视频检测
# ══════════════════════════════════════════════════════
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[YOLOv3] 视频信息: {width}×{height}, {fps_orig:.1f}fps, 共{total}帧")

    save_path = os.path.join(BASE_DIR, "yolo_result_video.avi")
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    writer    = cv2.VideoWriter(save_path, fourcc, fps_orig, (width, height))

    frame_id   = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        output, results, infer_time = detect_on_frame(frame)
        total_time += infer_time

        writer.write(output)

        if frame_id % 10 == 0:
            print(f"  帧 {frame_id}/{total}  推理={infer_time:.1f}ms  检测={len(results)}个目标")

        cv2.imshow("YOLOv3 - Video Detection  [按Q退出]", output)
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
    print("  YOLOv3 目标检测")
    print("=" * 50)

    print("\n[1/3] 检测图片: R-C.jpg")
    detect_image(IMAGE_PATH1)

    print("\n[2/3] 检测图片: R-C (1).jpg")
    detect_image(IMAGE_PATH2)

    print("\n[3/3] 检测视频: 5月22日.mp4")
    detect_video(VIDEO_PATH)