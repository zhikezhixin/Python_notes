import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

"""
实验二：VGG19实现图像分类
本程序做的事：
1. 加载VGG19预训练模型（ImageNet上训练好的权重）
2. 读取指定文件夹下的全部图片
3. 对每张图片做标准化预处理并前向推理
4. 输出Top-5预测类别和概率
5. 生成带预测结果标题的可视化图像，方便截图放入实验报告
"""


def load_model_and_metadata(device):
    # 加载 torchvision 自带的 VGG19 + 预训练权重
    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights)

    model.eval()
    model.to(device)

    # 类别名列表（ImageNet的1000个类别）
    idx_to_label = weights.meta["categories"]

    # 官方推荐的预处理变换（包含Resize/Crop/ToTensor/Normalize）
    preprocess_fn = weights.transforms()

    return model, idx_to_label, preprocess_fn


def preprocess_image(img_path, preprocess_fn, device):
    # 打开图像，并统一成 RGB（有的图是RGBA或灰度，必须转）
    img = Image.open(img_path).convert("RGB")

    # 做与ImageNet一致的预处理
    img_tensor = preprocess_fn(img)  # [3,224,224]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1,3,224,224]
    return img_tensor, img


def infer_single_image(model, img_tensor):
    # 前向推理 + softmax 概率
    with torch.no_grad():
        logits = model(img_tensor)          # [1,1000]
        prob = F.softmax(logits, dim=1)     # [1,1000]
    return prob.squeeze(0)  # [1000]


def get_topk_predictions(prob, idx_to_label, k=5):
    # 取Top-k预测
    topk_scores, topk_indices = torch.topk(prob, k)
    results = []
    for score, idx in zip(topk_scores, topk_indices):
        label = idx_to_label[int(idx)]
        results.append((label, float(score)))
    return results


def visualize_result(pil_img, top1_label, top1_score, save_path=None):
    # 画图，并把Top-1的分类结果写在标题上
    plt.figure()
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(f"预测: {top1_label} ({top1_score:.2f})")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[可视化已保存] {save_path}")
        plt.close()
    else:
        plt.show()


def classify_folder(img_dir, device, save_vis=True):
    """
    遍历文件夹 img_dir 下的所有图片，逐一分类。
    img_dir: 你的测试图片所在目录
    save_vis: 是否把带预测标题的结果图保存到output_vis文件夹
    """
    model, idx_to_label, preprocess_fn = load_model_and_metadata(device)

    # 确保输出结果目录存在
    os.makedirs("output_vis", exist_ok=True)

    # 遍历该路径下所有符合后缀的文件
    for filename in os.listdir(img_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(img_dir, filename)
        print("=" * 80)
        print(f"处理图片: {img_path}")

        # 1. 预处理
        img_tensor, pil_img = preprocess_image(img_path, preprocess_fn, device)

        # 2. 前向推理
        prob = infer_single_image(model, img_tensor)

        # 3. Top-5预测
        top5 = get_topk_predictions(prob, idx_to_label, k=5)

        # 4. 打印Top-5
        print("Top-5 预测结果:")
        for rank, (label, score) in enumerate(top5, start=1):
            print(f"  {rank}. {label:30s} 概率={score:.4f}")

        # 5. 可视化/保存
        top1_label, top1_score = top5[0]
        vis_path = os.path.join(
            "output_vis",
            f"{os.path.splitext(filename)[0]}_pred.png"
        ) if save_vis else None

        visualize_result(pil_img, top1_label, top1_score, save_path=vis_path)

        # 6. 给出一句可写进报告的解释
        print("说明：模型认为该图像最可能是:",
              top1_label,
              f"，置信度约 {top1_score:.2f}。")
        print("如果该类别与真实内容一致，说明模型识别正确；")
        print("若不一致，说明该物体可能不在ImageNet标准类别或出现了相似类别混淆。")

    print("=" * 80)
    print("推理完成。请用终端输出 + output_vis 下的图片作为实验截图。")


if __name__ == "__main__":
    # 自动选择 GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 使用设备:", device)

    # 你的实验图片目录（注意前面加 r 表示原始字符串，避免 \ 被当转义）
    img_dir = r"D:\Zeker\Documents\智能计算系统\实验二测试图片集"

    classify_folder(img_dir=img_dir, device=device, save_vis=True)
