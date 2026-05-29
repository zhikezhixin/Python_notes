import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 全局路径配置
# ============================================================
DATASET_ROOT = r"D:\Zeker\Documents\Computer vision\Experiment_4\dataset"

# Haar级联分类器XML文件所在目录（opencv自动定位）
HAAR_DIR = cv2.data.haarcascades

# 4种正脸分类器文件名
FACE_CASCADES = {
    "default":   "haarcascade_frontalface_default.xml",
    "alt":       "haarcascade_frontalface_alt.xml",
    "alt2":      "haarcascade_frontalface_alt2.xml",
    "alt_tree":  "haarcascade_frontalface_alt_tree.xml",
}

# 眼睛分类器
EYE_CASCADE_FILE = "haarcascade_eye.xml"

# 数据集：人名标签映射
PERSON_LABELS = {
    "person1": (0, "鞠婧祎"),
    "person2": (1, "刘浩存"),
    "person3": (2, "吴亦凡"),
    "person4": (3, "李易峰"),
}

# 支持的图像扩展名
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# ============================================================
# Part 0：工具函数
# ============================================================

def load_cascade(filename: str) -> cv2.CascadeClassifier:
    """加载Haar级联分类器"""
    path = os.path.join(HAAR_DIR, filename)
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise FileNotFoundError(f"无法加载级联分类器：{path}")
    return clf


def detect_faces(gray: np.ndarray,
                 cascade: cv2.CascadeClassifier,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: tuple = (30, 30)):
    """
    在灰度图中检测人脸，返回检测框列表。
    参数：
        gray         - 输入灰度图
        cascade      - 级联分类器
        scale_factor - 图像缩放比例（越大越快但可能漏检）
        min_neighbors- 每个候选框需要保留的最小邻近数（越大误检越少）
        min_size     - 最小检测窗口尺寸
    返回：
        faces - 检测到的人脸矩形列表 [(x, y, w, h), ...]
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces if len(faces) > 0 else []


def draw_faces(img: np.ndarray, faces, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """在彩色图上绘制人脸矩形框"""
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR（OpenCV默认）转 RGB（Matplotlib显示用）"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imread_unicode(path: str) -> np.ndarray:
    """支持中文路径读取图像"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ============================================================
# Part 1：单张图像人脸 + 眼睛检测
# ============================================================

def demo_face_eye_detection(img_path: str):
    """
    对单张图像进行人脸检测和眼睛检测，并可视化结果。
    步骤：
      1. 读取图像并转为灰度图
      2. 使用 default 分类器检测人脸
      3. 在每个人脸区域内使用眼睛分类器检测眼睛
      4. 绘制检测框并显示
    """
    img = imread_unicode(img_path)
    if img is None:
        print(f"[错误] 无法读取图像：{img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载分类器
    face_cascade = load_cascade(FACE_CASCADES["default"])
    eye_cascade  = load_cascade(EYE_CASCADE_FILE)

    # 检测人脸
    faces = detect_faces(gray, face_cascade)
    result = img.copy()

    for (x, y, w, h) in faces:
        # 绘制人脸框（绿色）
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 在人脸ROI中检测眼睛
        roi_gray = gray[y:y + h, x:x + w]
        eyes = detect_faces(roi_gray, eye_cascade,
                            scale_factor=1.1, min_neighbors=5, min_size=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            # 绘制眼睛框（蓝色），坐标要映射回原图
            cv2.rectangle(result,
                          (x + ex, y + ey),
                          (x + ex + ew, y + ey + eh),
                          (255, 0, 0), 2)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    axes[1].imshow(bgr_to_rgb(result))
    axes[1].set_title(f"人脸检测结果（检测到 {len(faces)} 张人脸）")
    axes[1].axis('off')
    plt.suptitle("人脸 & 眼睛检测", fontsize=14)
    plt.tight_layout()
    plt.show()
    print(f"[Part1] 检测到 {len(faces)} 张人脸")


# ============================================================
# Part 2：四种分类器性能对比（多场景）          ← 修复1
# ============================================================

def compare_cascades(dataset_root: str):
    """
    在多张不同场景图片上对比4种 frontalface 分类器的检测效果。
        函数签名改为接收 dataset_root，内部从每个人物文件夹各取1张图，
        共4张不同场景，对每张图分别运行4种分类器，输出
        4行(场景) × 4列(分类器) 的可视化对比网格，并附汇总统计表。
    """
    # 每个人物文件夹各取第一张图作为代表场景，共4张不同背景/角度的图
    scene_imgs  = []
    scene_names = []
    for folder, (_, person_name) in PERSON_LABELS.items():
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(IMG_EXTS):
                scene_imgs.append(os.path.join(folder_path, fname))
                scene_names.append(f"{person_name}/{fname}")
                break   # 每个人物只取第一张

    n_scenes   = len(scene_imgs)
    cnames     = list(FACE_CASCADES.keys())   # 4种分类器名称列表
    n_cascades = len(cnames)

    # stats[row][cascade_name] = 该场景该分类器检测到的人脸框数
    stats = [{} for _ in range(n_scenes)]

    fig, axes = plt.subplots(n_scenes, n_cascades,
                             figsize=(n_cascades * 5, n_scenes * 4))
    # 保证 axes 始终是二维结构，防止只有1行/1列时退化为一维
    if n_scenes == 1:
        axes = [axes]
    if n_cascades == 1:
        axes = [[ax] for ax in axes]

    for row, (img_path, scene_name) in enumerate(zip(scene_imgs, scene_names)):
        img = imread_unicode(img_path)
        if img is None:
            print(f"  [警告] 无法读取：{img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for col, cname in enumerate(cnames):
            cascade = load_cascade(FACE_CASCADES[cname])
            faces   = detect_faces(gray, cascade,
                                   scale_factor=1.1, min_neighbors=5)
            result  = draw_faces(img, faces)
            n_det   = len(faces)
            stats[row][cname] = n_det

            axes[row][col].imshow(bgr_to_rgb(result))
            # 第一行加粗显示分类器名作为列标题
            if row == 0:
                axes[row][col].set_title(f"【{cname}】\n检测框: {n_det}",
                                         fontsize=9, fontweight='bold')
            else:
                axes[row][col].set_title(f"检测框: {n_det}", fontsize=9)
            # 第一列显示人物/场景名作为行标签
            if col == 0:
                axes[row][col].set_ylabel(scene_name, fontsize=8,
                                          rotation=0, labelpad=80, va='center')
            axes[row][col].axis('off')

    plt.suptitle("四种 Haar 分类器 × 多场景 检测效果对比", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ── 控制台汇总统计表 ─────────────────────────────────────────
    print("\n[Part2] 各分类器 × 各场景 检测框数量汇总：")
    header = f"{'场景':<22}" + "".join(f"{c:>12}" for c in cnames)
    print("  " + header)
    print("  " + "-" * len(header))
    for row, sname in enumerate(scene_names):
        row_str = f"{sname:<22}" + "".join(
            f"{stats[row].get(c, '-'):>12}" for c in cnames)
        print("  " + row_str)

    print("\n  各分类器跨场景总检测框数（真实人脸数=4，越接近4越好）：")
    for cname in cnames:
        total = sum(s.get(cname, 0) for s in stats)
        print(f"    {cname:12s}: 共 {total} 框")

    return stats


# ============================================================
# Part 3：构建人脸数据集（读取 + 人脸裁剪 + 标签）
# ============================================================

def build_dataset(dataset_root: str, cascade_name: str = "default"):
    """
    遍历 dataset 目录，对每张图片进行人脸检测，
    将检测到的第一张人脸裁剪为 200x200 灰度图，
    与对应标签一起保存到列表。

    返回：
        faces  - list of np.ndarray, 每元素为 200x200 灰度人脸图
        labels - list of int, 人名标签
        names  - dict {label_id: 人名}
    """
    face_cascade = load_cascade(FACE_CASCADES[cascade_name])
    faces, labels = [], []
    names = {v[0]: v[1] for v in PERSON_LABELS.values()}

    print("\n[Part3] 开始构建数据集...")
    for folder, (label_id, person_name) in PERSON_LABELS.items():
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            print(f"  [警告] 目录不存在：{folder_path}")
            continue

        # 遍历该人物文件夹中所有图像
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(folder_path, fname)
            img  = imread_unicode(img_path)
            if img is None:
                print(f"  [警告] 无法读取：{img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = detect_faces(gray, face_cascade,
                                    scale_factor=1.1, min_neighbors=3)

            if len(detected) == 0:
                # 如果检测失败，直接使用全图缩放（降级处理）
                face_gray = cv2.resize(gray, (200, 200))
                print(f"  [降级] {fname}: 未检测到人脸，使用全图")
            else:
                # 取置信度最高的人脸（面积最大的框）
                (x, y, w, h) = max(detected, key=lambda r: r[2] * r[3])
                face_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            faces.append(face_gray)
            labels.append(label_id)
            print(f"  [OK] {person_name} / {fname} → label={label_id}")

    print(f"\n[Part3] 数据集构建完成：共 {len(faces)} 张人脸样本")
    return faces, labels, names


def visualize_dataset(faces: list, labels: list, names: dict):
    """可视化数据集中的人脸样本（每人显示最多4张）"""
    unique_labels = sorted(set(labels))
    n_persons = len(unique_labels)
    cols = 4
    fig, axes = plt.subplots(n_persons, cols, figsize=(cols * 3, n_persons * 3))

    for row, lbl in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels) if l == lbl]
        for col in range(cols):
            ax = axes[row][col] if n_persons > 1 else axes[col]
            if col < len(indices):
                ax.imshow(faces[indices[col]], cmap='gray')
                if col == 0:
                    ax.set_ylabel(names[lbl], fontsize=10, rotation=0,
                                  labelpad=60, va='center')
            ax.axis('off')

    plt.suptitle("人脸数据集预览（200×200 灰度）", fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================================================
# Part 4：LBPH 人脸识别 —— 留一法评估
# ============================================================

def train_lbph_recognizer(train_faces: list, train_labels: list):
    """
    使用 LBPH 算法训练人脸识别模型。
    LBPH 原理：
      对每个像素，将其与周围 P 个邻域像素比较，
      若中心像素 >= 邻域像素则编码为1，否则为0，
      形成一个 P 位二进制数，统计其直方图作为特征描述子。
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,        # LBP采样半径
        neighbors=8,     # 采样点数
        grid_x=8,        # 将人脸分为 8×8 的网格
        grid_y=8,
        threshold=200.0  # 置信度阈值，超过此值认为"未知"
    )
    recognizer.train(train_faces, np.array(train_labels))
    return recognizer


def leave_one_out_evaluation(faces: list, labels: list, names: dict):
    """
    留一法（Leave-One-Out）交叉验证：
    每次取一张作为测试样本，其余全部用于训练，
    统计总体识别准确率和每人准确率。
    """
    n = len(faces)
    correct = 0
    per_person_correct = {lbl: [0, 0] for lbl in set(labels)}  # [正确, 总数]
    results_log = []

    print("\n[Part4] LBPH 留一法评估开始...")
    for test_idx in range(n):
        # 构造训练集（排除当前测试样本）
        train_faces  = [faces[i]  for i in range(n) if i != test_idx]
        train_labels = [labels[i] for i in range(n) if i != test_idx]

        # 训练识别器
        recognizer = train_lbph_recognizer(train_faces, train_labels)

        # 预测
        pred_label, confidence = recognizer.predict(faces[test_idx])
        true_label = labels[test_idx]
        is_correct = (pred_label == true_label)

        if is_correct:
            correct += 1
        per_person_correct[true_label][0] += is_correct
        per_person_correct[true_label][1] += 1

        results_log.append({
            "idx": test_idx,
            "true": true_label,
            "pred": pred_label,
            "conf": confidence,
            "ok":   is_correct
        })
        print(f"  样本{test_idx:02d} 真实:{names[true_label]:4s} "
              f"预测:{names.get(pred_label, '未知'):4s} "
              f"置信度:{confidence:.1f}  {'✓' if is_correct else '✗'}")

    # 统计结果
    accuracy = correct / n * 100
    print(f"\n[Part4] 总体识别准确率：{correct}/{n} = {accuracy:.1f}%")
    print("\n[Part4] 各人识别准确率：")
    for lbl, (c, t) in per_person_correct.items():
        print(f"  {names[lbl]}：{c}/{t} = {c/t*100:.1f}%")

    return results_log, accuracy


def visualize_recognition_results(faces: list, results_log: list, names: dict):
    """可视化每个测试样本的预测结果（绿色=正确，红色=错误）"""
    n = len(results_log)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = axes.flatten()

    for i, res in enumerate(results_log):
        ax = axes[i]
        ax.imshow(faces[res["idx"]], cmap='gray')
        true_name = names[res["true"]]
        pred_name = names.get(res["pred"], "未知")
        color = "green" if res["ok"] else "red"
        ax.set_title(f"真:{true_name}\n预:{pred_name}\n({res['conf']:.0f})",
                     fontsize=7, color=color)
        ax.axis('off')

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle("LBPH 人脸识别结果（绿=正确 / 红=错误）", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# Part 5A：视频文件人脸检测 + LBPH 人脸识别
# ============================================================

# 标注框颜色表（每个人物分配固定颜色，BGR格式）
LABEL_COLORS = {
    0: (0,   200,  50),   # 鞠婧祎 → 绿色
    1: (255, 180,   0),   # 刘浩存 → 黄色
    2: (0,   100, 255),   # 吴亦凡 → 橙色
    3: (180,   0, 255),   # 李易峰 → 紫色
}
UNKNOWN_COLOR = (100, 100, 100)  # 未知人脸 → 灰色

# 人名 → 视频显示用英文缩写映射
LABEL_ABBR = {
    0: "JJY",   # 鞠婧祎
    1: "LHC",   # 刘浩存
    2: "WYF",   # 吴亦凡
    3: "LYF",   # 李易峰
}


def video_face_recognition(video_source,
                            recognizer,
                            names: dict,
                            cascade_name: str = "default",
                            conf_threshold: float = 130.0,
                            scale: float = 1.0,
                            save_output: bool = False,
                            output_path: str = "output_recognition.avi"):
    """
    对视频文件或摄像头进行逐帧人脸检测 + LBPH识别，实时标注人名。

    参数：
        video_source   - 视频文件路径（字符串）或摄像头索引（整数，如 0）
        recognizer     - 已训练好的 LBPH 识别器
        names          - {label_id: 人名} 字典
        cascade_name   - 使用的 Haar 分类器名称
        conf_threshold - 置信度阈值：LBPH置信度越低越好，超过此值标为"Unknown"
        scale          - 显示窗口缩放比例（如0.5表示缩小到一半，便于大分辨率视频）
        save_output    - 是否将结果保存为视频文件
        output_path    - 输出视频保存路径

    操作说明：
        按 'q' 或 ESC 退出；按 空格 暂停/继续；按 's' 截图保存当前帧。

    流程：
        1. 读取每一帧
        2. 转灰度 → Haar检测人脸框
        3. 裁剪人脸ROI → 缩放至200×200 → LBPH预测
        4. 根据置信度决定显示人名还是"Unknown"
        5. 在原帧上绘制彩色矩形框 + 标签 + 置信度
    """
    is_file = isinstance(video_source, str)
    source_label = os.path.basename(video_source) if is_file else f"摄像头[{video_source}]"

    # 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源：{video_source}")
        return

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else -1
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_file:
        print(f"\n[Part5A] 视频文件：{source_label}")
        print(f"         分辨率：{width}×{height}  帧率：{fps:.1f}fps  "
              f"总帧数：{total_frames}")
    else:
        print(f"\n[Part5A] 摄像头实时识别已启动")

    # 准备视频输出（可选）
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_w  = int(width  * scale)
        out_h  = int(height * scale)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        print(f"         输出保存至：{output_path}")

    face_cascade = load_cascade(FACE_CASCADES[cascade_name])

    frame_idx  = 0
    paused     = False
    screenshot = 0
    print("         操作：[q/ESC] 退出  [空格] 暂停/继续  [s] 截图")
    print("-" * 55)

    while True:
        # ── 暂停逻辑 ────────────────────────────────────────────
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = False
            continue

        ret, frame = cap.read()
        if not ret:
            if is_file:
                print("[Part5A] 视频播放完毕")
            break

        frame_idx += 1
        display = frame.copy()

        # ── 人脸检测 ─────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 对大分辨率视频先缩小灰度图检测，加速处理
        detect_gray = gray
        if width > 1280:
            detect_gray = cv2.resize(gray, (width // 2, height // 2))

        raw_faces = detect_faces(detect_gray, face_cascade,
                                 scale_factor=1.1, min_neighbors=4,
                                 min_size=(40, 40))

        # 如果检测时缩小了，坐标要还原
        scale_back = 2 if width > 1280 else 1

        # ── 逐个人脸识别并标注 ───────────────────────────────────
        for (x, y, w, h) in raw_faces:
            x, y, w, h = x * scale_back, y * scale_back, w * scale_back, h * scale_back

            # 裁剪人脸ROI，缩放到200×200与训练时一致
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, (200, 200))

            # LBPH 识别
            pred_label, confidence = recognizer.predict(roi_resized)

            # 置信度判断：LBPH中值越小越相似
            if confidence < conf_threshold:
                person_name = names.get(pred_label, "?")
                abbr        = LABEL_ABBR.get(pred_label, "?")
                box_color   = LABEL_COLORS.get(pred_label, UNKNOWN_COLOR)
                label_text  = f"{abbr} [{confidence:.1f}]"
            else:
                person_name = "Unknown"
                abbr        = "UNK"
                box_color   = UNKNOWN_COLOR
                label_text  = f"Unknown [{confidence:.1f}]"

            # 绘制人脸矩形框
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)

            # 绘制标签背景色块（提高文字可读性）
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            (tw, th), _ = cv2.getTextSize(label_text,
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(display,
                          (x, label_y - th - 4),
                          (x + tw + 4, label_y + 2),
                          box_color, -1)  # 实心填充

            # 绘制白色文字
            cv2.putText(display, label_text, (x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # ── 右上角显示帧号 / 进度 ────────────────────────────────
        if is_file and total_frames > 0:
            progress = f"Frame {frame_idx}/{total_frames}  ({frame_idx/total_frames*100:.1f}%)"
        else:
            progress = f"Frame {frame_idx}"
        cv2.putText(display, progress, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 显示窗口 ──────────────────────────────────────────────
        if scale != 1.0:
            show = cv2.resize(display, (int(width * scale), int(height * scale)))
        else:
            show = display

        cv2.imshow(f"人脸检测识别 — {source_label}  (q=退出 空格=暂停)", show)

        # 保存帧到输出视频
        if writer is not None:
            writer.write(show)

        # ── 按键响应 ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):       # q 或 ESC 退出
            break
        elif key == ord(' '):           # 空格 暂停
            paused = True
            print(f"  [暂停] 第 {frame_idx} 帧，按空格继续...")
        elif key == ord('s'):           # s 截图
            screenshot += 1
            shot_path = f"screenshot_{screenshot:03d}.jpg"
            cv2.imwrite(shot_path, display)
            print(f"  [截图] 已保存：{shot_path}")

    # ── 收尾 ──────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[Part5A] 结束，共处理 {frame_idx} 帧")


# ============================================================
# Part 5B：摄像头实时人脸检测（纯检测，不含识别）
# ============================================================

def realtime_face_detection(cascade_name: str = "default"):
    """
    从摄像头采集视频流，实时进行人脸检测（不含识别）。
    按 'q' 键退出。
    """
    face_cascade = load_cascade(FACE_CASCADES[cascade_name])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    print("[Part5B] 摄像头人脸检测已启动，按 'q' 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, face_cascade)
        result = draw_faces(frame, faces)

        cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("实时人脸检测 (按 q 退出)", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Part5B] 实时检测已退出")


# ============================================================
# Part 6：EigenFaces / FisherFaces / LBPH 三种方法对比  ← 修复2
# ============================================================

def compare_recognizers(faces: list, labels: list, names: dict):
    """
    对比 EigenFaces、FisherFaces、LBPH 三种方法的留一法准确率。


    修复方案：
        为每种方法单独维护 tested 计数器，仅在成功完成预测后
        才将 tested +1，最终以 correct/tested 为准计算准确率，
        并在打印时注明跳过了多少样本。
    """
    n = len(faces)

    methods = {
        "LBPH":        lambda: cv2.face.LBPHFaceRecognizer_create(
                           radius=1, neighbors=8, grid_x=8, grid_y=8,
                           threshold=9999.0),
        "EigenFaces":  lambda: cv2.face.EigenFaceRecognizer_create(
                           num_components=0, threshold=9999.0),
        "FisherFaces": lambda: cv2.face.FisherFaceRecognizer_create(
                           num_components=0, threshold=9999.0),
    }

    print("\n[Part6] 三种识别方法准确率对比（留一法）：")
    comparison = {}

    for method_name, creator in methods.items():
        correct = 0
        tested  = 0   # ← 修复核心：独立记录实际参与评估的样本数

        for test_idx in range(n):
            train_faces  = [faces[i]  for i in range(n) if i != test_idx]
            train_labels = [labels[i] for i in range(n) if i != test_idx]

            # FisherFaces 约束：训练集中每类样本数必须 >= 2
            # 本数据集每人4张，留一后最少剩3张，正常不会触发；
            # 此处保留检查以保证代码在更小数据集上的鲁棒性。
            if method_name == "FisherFaces":
                label_counts = {}
                for lbl in train_labels:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                if any(v < 2 for v in label_counts.values()):
                    continue   # 跳过本轮，tested 不增加 ← 修复关键

            try:
                rec = creator()
                rec.train(train_faces, np.array(train_labels))
                pred, _ = rec.predict(faces[test_idx])
                tested += 1                   # 成功预测才计入分母
                if pred == labels[test_idx]:
                    correct += 1
            except cv2.error as e:
                print(f"  [{method_name}] 跳过样本{test_idx}: {e}")

        # 以 tested（而非 n）为分母，准确率计算公平准确
        acc = (correct / tested * 100) if tested > 0 else 0.0
        comparison[method_name] = acc

        skip_note = (f"  （{n - tested} 个样本因约束跳过，未计入分母）"
                     if tested < n else "")
        print(f"  {method_name:12s}: {correct}/{tested} = {acc:.1f}%{skip_note}")

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(comparison.keys()), list(comparison.values()),
                  color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    ax.set_ylabel("识别准确率 (%)")
    ax.set_title("EigenFaces / FisherFaces / LBPH 准确率对比（留一法）")
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, comparison.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha='center', fontsize=11)
    plt.tight_layout()
    plt.show()

    return comparison


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":

    # ── Part 1：单张图像人脸 + 眼睛检测 ─────────────────────────
    demo_img = os.path.join(DATASET_ROOT, "person1", "jjy1.jpg")
    print("=" * 60)
    print("Part 1：人脸 & 眼睛检测演示")
    print("=" * 60)
    demo_face_eye_detection(demo_img)

    # ── Part 2：四种分类器 × 多场景 效果对比 ────────────────────
    # 【修复1】传入 DATASET_ROOT 而非单张图路径，函数内部自动从
    # 4个人物文件夹各取1张图，覆盖4种不同场景，对比更具说服力
    print("\n" + "=" * 60)
    print("Part 2：四种 Haar 分类器多场景对比")
    print("=" * 60)
    compare_cascades(DATASET_ROOT)

    # ── Part 3：构建数据集 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 3：构建人脸数据集")
    print("=" * 60)
    faces, labels, names = build_dataset(DATASET_ROOT)
    visualize_dataset(faces, labels, names)

    # ── Part 4：LBPH 人脸识别（留一法）──────────────────────────
    print("\n" + "=" * 60)
    print("Part 4：LBPH 人脸识别（留一法评估）")
    print("=" * 60)
    results_log, accuracy = leave_one_out_evaluation(faces, labels, names)
    visualize_recognition_results(faces, results_log, names)

    # ── Part 5A：视频文件人脸检测 + LBPH 识别 ───────────────────
    print("\n" + "=" * 60)
    print("Part 5A：视频文件人脸检测 + 识别")
    print("=" * 60)
    VIDEO_PATH = r"D:\Zeker\Documents\Computer vision\Experiment_4\dataset\lhc.mp4"
    final_recognizer = train_lbph_recognizer(faces, labels)   # 全量数据训练
    video_face_recognition(
        video_source   = VIDEO_PATH,
        recognizer     = final_recognizer,
        names          = names,
        cascade_name   = "default",
        conf_threshold = 130.0,
        scale          = 0.75,
        save_output    = True,
        output_path    = r"D:\Zeker\Documents\Computer vision\Experiment_4\lhc_result.avi"
    )

    # ── Part 5B：摄像头实时人脸检测（可选，取消注释即可）───────
    print("\n" + "=" * 60)
    print("Part 5B：摄像头实时人脸检测")
    print("=" * 60)
    realtime_face_detection(cascade_name="default")

    # ── Part 6：三种方法对比（留一法，分母已修复）───────────────
    # 【修复2】compare_recognizers 内部现在以 tested 为分母，
    # FisherFaces 跳过的轮次不再错误计入分母，三种方法准确率公平可比
    print("\n" + "=" * 60)
    print("Part 6：三种识别方法准确率对比")
    print("=" * 60)
    compare_recognizers(faces, labels, names)

    print("\n全部实验完成！")
