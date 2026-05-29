import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 中文
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 仅使用你本地三张图片 ==========
IMAGE_PATHS = [
    r"D:\Zeker\Documents\智能计算系统\实验三数据集\室内静物.jpg",
    r"D:\Zeker\Documents\智能计算系统\实验三数据集\城市风景.jpg",
    r"D:\Zeker\Documents\智能计算系统\实验三数据集\人像.jpg",
]

# 可选：若你有本地训练好的权重（*.pth），填这里；留空则用随机权重（仅流程/计时）
WEIGHTS_PATH = r""  # 例：r"D:\Zeker\Documents\智能计算系统\实验三数据集\mosaic.pth"

# ==================== 网络定义 ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + r

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样
        self.conv1 = nn.Conv2d(3, 32, 9, 1, 4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # 残差
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # 上采样
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn.Conv2d(32, 3, 9, 1, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.res1(out); out = self.res2(out); out = self.res3(out); out = self.res4(out); out = self.res5(out)
        out = self.relu(self.in4(self.deconv1(out)))
        out = self.relu(self.in5(self.deconv2(out)))
        out = self.deconv3(out)
        return out

# ==================== 工具函数 ====================
to_tensor_255 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255.0))
])

def load_image_as_tensor(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片: {path}")
    img = Image.open(path).convert('RGB')
    t = to_tensor_255(img).unsqueeze(0).to(device)
    return img, t

@torch.no_grad()
def tensor_to_pil_255(t):
    t = t.clamp(0, 255).squeeze(0).cpu() / 255.0
    return transforms.ToPILImage()(t)

def measure_performance(model, image_tensor, device, warmup=3, num_runs=10):
    """统一的推断计时（含预热 & 同步）"""
    model.eval()
    x = image_tensor.to(device)

    # 预热
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 计时
    times, y = [], None
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    return {
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'fps': 1.0 / float(np.mean(times)),
        'output': y
    }

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def basename_noext(p): return os.path.splitext(os.path.basename(p))[0]

# 尝试 FX 静态量化；失败则退回动态量化占位
def build_int8_model(model_fp32, calib_tensors):
    try:
        import torch.ao.quantization as tq
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        torch.backends.quantized.engine = "fbgemm"
        qconfig = tq.get_default_qconfig("fbgemm")
        qmap = tq.QConfigMapping().set_global(qconfig)
        # 跳过不友好模块
        qmap = qmap.set_object_type(nn.InstanceNorm2d, None)
        qmap = qmap.set_object_type(nn.ConvTranspose2d, None)

        example = calib_tensors[0]
        prepared = prepare_fx(model_fp32.eval(), qmap, example)
        with torch.no_grad():
            for t in calib_tensors:
                _ = prepared(t)
        int8_model = convert_fx(prepared).eval()
        print("✓ FX 静态量化成功（Conv 部分 INT8）")
        return int8_model, "fx_static"
    except Exception as e:
        print(f"⚠ FX 静态量化失败，退回动态量化占位：{e}")
        # 动态量化对 Conv 基本无效，这里仅占位以满足“量化流程”
        int8_model = torch.quantization.quantize_dynamic(model_fp32, {nn.Conv2d}, dtype=torch.qint8)
        return int8_model, "dynamic_placeholder"

# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print(" 仅使用本地三张图片：CPU / INT8 / GPU 推断与对比（不联网，不选风格）")
    print("=" * 80)

    # 1) 模型与权重
    model = TransformerNet().eval()
    style_tag = "untrained"
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location='cpu')
            model.load_state_dict(state, strict=True)
            style_tag = os.path.splitext(os.path.basename(WEIGHTS_PATH))[0]
            print(f"✓ 已加载本地权重：{WEIGHTS_PATH}")
        except Exception as e:
            print(f"⚠ 加载权重失败，将以随机权重继续（仅流程/计时）：{e}")
    else:
        print("⚠ 未提供有效权重，将以随机权重推断（结果仅作流程与计时展示）")

    # 2) 设备
    has_cuda = torch.cuda.is_available()
    device_cpu = torch.device('cpu')
    model_cpu = model.to(device_cpu)
    if has_cuda:
        device_gpu = torch.device('cuda')
        model_gpu = model.to(device_gpu)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print("\n【GPU信息】")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB | 计算能力: {props.major}.{props.minor}")
    else:
        print("\n未检测到 GPU：将仅进行 CPU/量化测试。")

    # 3) 读取三张图片
    print("\n【读取本地图片】")
    image_list, tensor_list = [], []
    for p in IMAGE_PATHS:
        img, ten = load_image_as_tensor(p, device_cpu)
        image_list.append((p, img))
        tensor_list.append(ten)
        print(f"  ✓ {p}")

    # 4) 构建 INT8 模型（优先 FX 静态量化）
    int8_model, quant_mode = build_int8_model(model_cpu, tensor_list)

    # 5) 跑实验
    out_root = os.path.join("outputs", style_tag)
    ensure_dir(out_root)
    all_results = []   # [(name, stats_cpu, stats_int8, stats_gpu or None)]

    for (p, src_img), x_cpu in zip(image_list, tensor_list):
        name = basename_noext(p)
        print("\n" + "-" * 80)
        print(f"处理图片：{p}")

        out_dir = os.path.join(out_root, name)
        ensure_dir(out_dir)

        # CPU(FP32)
        print("【CPU(FP32)】")
        stats_cpu = measure_performance(model_cpu, x_cpu, device_cpu)
        out_cpu = tensor_to_pil_255(stats_cpu['output'])
        out_cpu.save(os.path.join(out_dir, f"{style_tag}_{name}_cpu.jpg"))
        print(f"    平均时间: {stats_cpu['avg_time']:.4f}s | FPS: {stats_cpu['fps']:.2f}")

        # CPU(INT8)
        print(f"【CPU(INT8: {quant_mode})】")
        stats_i8 = measure_performance(int8_model, x_cpu, device_cpu)
        out_i8 = tensor_to_pil_255(stats_i8['output'])
        out_i8.save(os.path.join(out_dir, f"{style_tag}_{name}_cpu_int8.jpg"))
        print(f"    平均时间: {stats_i8['avg_time']:.4f}s | FPS: {stats_i8['fps']:.2f} | "
              f"加速比(相对FP32): {stats_cpu['avg_time']/stats_i8['avg_time']:.2f}x")

        # GPU(FP32, 如可用)
        if has_cuda:
            print("【GPU(FP32)】")
            stats_gpu = measure_performance(model_gpu, x_cpu, device_gpu)
            out_gpu = tensor_to_pil_255(stats_gpu['output'])
            out_gpu.save(os.path.join(out_dir, f"{style_tag}_{name}_gpu.jpg"))
            print(f"    平均时间: {stats_gpu['avg_time']:.4f}s | FPS: {stats_gpu['fps']:.2f} | "
                  f"GPU/CPU 加速比: {stats_cpu['avg_time']/stats_gpu['avg_time']:.2f}x")
        else:
            stats_gpu = None

        # 三图对比（输入/CPU/(GPU)）
        cols = 3 if has_cuda else 2
        fig, axes = plt.subplots(1, cols, figsize=(5.5 * cols, 5.5))
        if cols == 2:
            ax0, ax1 = axes
            ax0.imshow(src_img); ax0.set_title("输入"); ax0.axis('off')
            ax1.imshow(out_cpu); ax1.set_title(f"CPU FP32\n{stats_cpu['avg_time']:.4f}s"); ax1.axis('off')
        else:
            axes[0].imshow(src_img); axes[0].set_title("输入"); axes[0].axis('off')
            axes[1].imshow(out_cpu);  axes[1].set_title(f"CPU FP32\n{stats_cpu['avg_time']:.4f}s"); axes[1].axis('off')
            axes[2].imshow(out_gpu);  axes[2].set_title(f"GPU FP32\n{stats_gpu['avg_time']:.4f}s"); axes[2].axis('off')
        plt.suptitle(f"{style_tag} | {name}", fontweight='bold')
        cmp_path = os.path.join(out_dir, f"{style_tag}_{name}_comparison.png")
        plt.tight_layout(); plt.savefig(cmp_path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"✓ 已保存对比图：{cmp_path}")

        all_results.append((name, stats_cpu, stats_i8, stats_gpu))

    # 6) 汇总表打印
    print("\n" + "=" * 80)
    print(f"权重：{('未加载(随机)') if style_tag=='untrained' else WEIGHTS_PATH} | INT8 模式：{quant_mode}")
    header = f"{'图片':<16}{'CPU(s)':>10}{'CPU FPS':>10}{'INT8(s)':>10}{'INT8 FPS':>10}{'GPU(s)':>10}{'GPU FPS':>10}{'GPU/CPU':>10}"
    print(header)
    print("-" * len(header))
    for name, s_cpu, s_i8, s_gpu in all_results:
        gpu_t = f"{s_gpu['avg_time']:.4f}" if s_gpu else "-"
        gpu_f = f"{s_gpu['fps']:.1f}" if s_gpu else "-"
        speedup = f"{(s_cpu['avg_time']/s_gpu['avg_time']):.2f}x" if s_gpu else "-"
        print(f"{name:<16}{s_cpu['avg_time']:>10.4f}{s_cpu['fps']:>10.1f}"
              f"{s_i8['avg_time']:>10.4f}{s_i8['fps']:>10.1f}"
              f"{gpu_t:>10}{gpu_f:>10}{speedup:>10}")
    print("=" * 80)

    # 7) 生成总体性能汇总图（按平台平均）
    platforms = ["CPU_FP32", f"CPU_INT8({quant_mode})"] + (["GPU_FP32"] if has_cuda else [])
    avg_times = []
    for pid in platforms:
        times = []
        for _, s_cpu, s_i8, s_gpu in all_results:
            if pid.startswith("CPU_FP32"):
                times.append(s_cpu['avg_time'])
            elif pid.startswith("CPU_INT8"):
                times.append(s_i8['avg_time'])
            elif pid.startswith("GPU_FP32") and s_gpu:
                times.append(s_gpu['avg_time'])
        if times:
            avg_times.append(np.mean(times))
        else:
            avg_times.append(np.nan)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.bar(platforms, avg_times, edgecolor='black')
    ax.set_ylabel("平均推断时间 (秒)")
    ax.set_title("按平台的平均推断时间（3张图平均）")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for b, t in zip(bars, avg_times):
        if not np.isnan(t):
            ax.text(b.get_x()+b.get_width()/2, t, f"{t:.4f}s", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(out_root, f"{style_tag}_summary_times.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"✓ 已保存总体性能图：{summary_path}")

    print(f"\n所有结果已保存在：outputs\\{style_tag}\\<图片名>\\")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
