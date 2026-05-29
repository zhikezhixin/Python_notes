"""
对比脚本：在同一图片上运行两种模型并并排显示结果、输出性能对比表
"""

import cv2
import numpy as np
import os

# 导入两个检测模块的核心函数
from mobilenet_ssd_detect import detect_on_frame as ssd_detect
from yolo_detect           import detect_on_frame as yolo_detect

BASE_DIR    = r"D:\CodeManager\PycharmProejcts\Python_notes\Computer Vision\Experiment_5"
IMAGE_PATH1 = os.path.join(BASE_DIR, "R-C.jpg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "R-C (1).jpg")


def compare_on_image(image_path):
    """对同一张图片分别用两种模型检测，并排显示对比结果"""
    raw   = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[ERROR] 无法读取: {image_path}")
        return

    filename = os.path.basename(image_path)

    # 分别运行两种模型
    ssd_out,  ssd_results,  ssd_time  = ssd_detect(frame)
    yolo_out, yolo_results, yolo_time = yolo_detect(frame)

    # ── 控制台打印对比表格 ──────────────────────────────
    print(f"\n{'='*58}")
    print(f"  对比图片: {filename}")
    print(f"{'='*58}")
    print(f"{'指标':<20} {'MobileNet_SSD':>16} {'YOLOv3':>16}")
    print(f"{'-'*58}")
    print(f"{'推理时间(ms)':<20} {ssd_time:>14.1f}   {yolo_time:>14.1f}")
    print(f"{'检测目标数量':<20} {len(ssd_results):>14}   {len(yolo_results):>14}")
    print(f"{'支持类别数':<20} {'20 (VOC)':>14}   {'80 (COCO)':>14}")
    print(f"{'输入尺寸':<20} {'300×300':>14}   {'416×416':>14}")
    print(f"{'检测架构':<20} {'单阶段 SSD':>14}   {'单阶段 YOLO':>13}")
    print(f"{'基础网络':<20} {'MobileNet':>14}   {'Darknet-53':>14}")
    print(f"{'后处理 NMS':<20} {'内置':>14}   {'需要':>14}")
    print(f"{'='*58}")

    # 打印具体检测结果
    print(f"\n[MobileNet_SSD] 检测到 {len(ssd_results)} 个目标:")
    for (label, conf, box) in ssd_results:
        print(f"  ▸ {label:<15} 置信度={conf:.4f}")

    print(f"\n[YOLOv3] 检测到 {len(yolo_results)} 个目标:")
    for (label, conf, box) in yolo_results:
        print(f"  ▸ {label:<15} 置信度={conf:.4f}")

    # ── 并排可视化 ────────────────────────────────────
    # 统一图像高度
    target_h = max(ssd_out.shape[0], yolo_out.shape[0])

    def resize_h(img, th):
        ratio = th / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), th))

    ssd_out  = resize_h(ssd_out,  target_h)
    yolo_out = resize_h(yolo_out, target_h)

    # 添加标题栏
    def add_title_bar(img, title, sub):
        bar = np.zeros((50, img.shape[1], 3), dtype="uint8")
        cv2.putText(bar, title, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bar, sub, (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return np.vstack([bar, img])

    ssd_out  = add_title_bar(ssd_out,
                             "MobileNet_SSD (Caffe)",
                             f"t={ssd_time:.1f}ms  n={len(ssd_results)}")
    yolo_out = add_title_bar(yolo_out,
                             "YOLOv3 (Darknet)",
                             f"t={yolo_time:.1f}ms  n={len(yolo_results)}")

    # 加分隔线后水平拼接
    divider = np.zeros((ssd_out.shape[0], 4, 3), dtype="uint8")
    combined = np.hstack([ssd_out, divider, yolo_out])

    # 保存与显示
    save_name = f"compare_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    save_path = os.path.join(BASE_DIR, save_name)
    cv2.imencode('.jpg', combined)[1].tofile(save_path)
    print(f"\n[INFO] 对比图已保存: {save_path}")

    # 缩放以适应屏幕显示
    disp_w = min(combined.shape[1], 1600)
    ratio  = disp_w / combined.shape[1]
    disp   = cv2.resize(combined, (disp_w, int(combined.shape[0] * ratio)))
    cv2.imshow(f"对比: {filename}  [按任意键继续]", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 58)
    print("  目标检测性能对比：MobileNet_SSD  vs  YOLOv3")
    print("=" * 58)

    compare_on_image(IMAGE_PATH1)
    compare_on_image(IMAGE_PATH2)