import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 网络定义 ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # 下采样
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # 残差块
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # 上采样
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = self.relu(self.in4(self.deconv1(out)))
        out = self.relu(self.in5(self.deconv2(out)))
        out = self.deconv3(out)

        return out


# ==================== 性能测试函数 ====================
def measure_performance(model, image_tensor, device, warmup=3, num_runs=10):
    """精确的性能测试"""
    model.eval()
    image_tensor = image_tensor.to(device)

    # 预热
    print(f"    预热 {warmup} 次...", end='', flush=True)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(image_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(" 完成")

    # 测试
    print(f"    测试 {num_runs} 次...", end='', flush=True)
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            output = model(image_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times.append(time.time() - start)
    print(" 完成")

    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'output': output
    }


# ==================== 主实验 ====================
def main():
    print("=" * 80)
    print(" " * 25 + "实时风格迁移完整实验")
    print(" " * 20 + "RTX 4060 Laptop GPU 版本")
    print("=" * 80)

    # 显示GPU信息
    print("\n【GPU信息】")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"  总显存: {props.total_memory / 1024 ** 3:.2f} GB")
    print(f"  计算能力: {props.major}.{props.minor}")

    # 启用优化
    torch.backends.cudnn.benchmark = True
    print("  ✓ 已启用cuDNN自动优化")

    # 准备测试图像
    print("\n【准备测试数据】")
    test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    test_image.save('input_image.jpg')
    print("  ✓ 测试图像已创建: input_image.jpg (512×512)")

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image_tensor = transform(test_image).unsqueeze(0)

    # 初始化模型
    print("\n【初始化模型】")
    model = TransformerNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    results = {}

    # ==================== CPU测试 ====================
    print("\n" + "=" * 80)
    print("【weather_scraper.py】CPU (FP32) 推断")
    print("=" * 80)

    device_cpu = torch.device('cpu')
    model_cpu = model.to(device_cpu)

    stats_cpu = measure_performance(model_cpu, image_tensor, device_cpu)
    results['CPU (FP32)'] = stats_cpu

    print(f"\n  结果:")
    print(f"    平均时间: {stats_cpu['avg_time']:.4f} ± {stats_cpu['std_time']:.4f} 秒")
    print(f"    最快时间: {stats_cpu['min_time']:.4f} 秒")
    print(f"    吞吐量:   {stats_cpu['fps']:.2f} FPS")

    # 保存输出
    output_cpu = stats_cpu['output'].squeeze(0).cpu().clamp(0, 255)
    output_image_cpu = transforms.ToPILImage()(output_cpu.div(255))
    output_image_cpu.save('output_cpu.jpg')
    print(f"    ✓ 输出已保存: output_cpu.jpg")

    # ==================== CPU量化测试 ====================
    print("\n" + "=" * 80)
    print("【实验2】CPU (INT8量化) 推断")
    print("=" * 80)

    print("  量化模型...")
    model_quantized = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Conv2d, nn.ConvTranspose2d}, dtype=torch.qint8
    )
    print("  ✓ 量化完成")

    stats_quant = measure_performance(model_quantized, image_tensor, device_cpu)
    results['CPU (INT8)'] = stats_quant

    print(f"\n  结果:")
    print(f"    平均时间: {stats_quant['avg_time']:.4f} ± {stats_quant['std_time']:.4f} 秒")
    print(f"    最快时间: {stats_quant['min_time']:.4f} 秒")
    print(f"    吞吐量:   {stats_quant['fps']:.2f} FPS")
    print(f"    加速比:   {stats_cpu['avg_time'] / stats_quant['avg_time']:.2f}x")

    output_quant = stats_quant['output'].squeeze(0).cpu().clamp(0, 255)
    output_image_quant = transforms.ToPILImage()(output_quant.div(255))
    output_image_quant.save('output_quantized.jpg')
    print(f"    ✓ 输出已保存: output_quantized.jpg")

    # ==================== GPU测试 ====================
    print("\n" + "=" * 80)
    print("【实验3】GPU (RTX 4060) 推断")
    print("=" * 80)

    device_gpu = torch.device('cuda')
    model_gpu = model.to(device_gpu)

    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    stats_gpu = measure_performance(model_gpu, image_tensor, device_gpu)
    results['GPU (RTX 4060)'] = stats_gpu

    # 显存使用
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    memory_peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f"\n  结果:")
    print(f"    平均时间: {stats_gpu['avg_time']:.4f} ± {stats_gpu['std_time']:.4f} 秒")
    print(f"    最快时间: {stats_gpu['min_time']:.4f} 秒")
    print(f"    吞吐量:   {stats_gpu['fps']:.2f} FPS")
    print(f"    显存使用: {memory_allocated:.2f} MB (当前)")
    print(f"    显存峰值: {memory_peak:.2f} MB")
    print(f"    相对CPU加速: {stats_cpu['avg_time'] / stats_gpu['avg_time']:.2f}x")
    print(f"    相对量化CPU加速: {stats_quant['avg_time'] / stats_gpu['avg_time']:.2f}x")

    output_gpu = stats_gpu['output'].squeeze(0).cpu().clamp(0, 255)
    output_image_gpu = transforms.ToPILImage()(output_gpu.div(255))
    output_image_gpu.save('output_gpu.jpg')
    print(f"    ✓ 输出已保存: output_gpu.jpg")

    # ==================== 批处理测试 ====================
    print("\n" + "=" * 80)
    print("【实验4】GPU批处理性能测试")
    print("=" * 80)

    batch_sizes = [1, 2, 4, 8]
    print(f"\n  {'批大小':<10} {'总时间(s)':<12} {'单张(s)':<12} {'吞吐量(FPS)':<12}")
    print("  " + "-" * 50)

    for bs in batch_sizes:
        try:
            batch_input = torch.randn(bs, 3, 512, 512).to(device_gpu)

            # 预热
            with torch.no_grad():
                _ = model_gpu(batch_input)
            torch.cuda.synchronize()

            # 测试
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model_gpu(batch_input)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = np.mean(times)
            per_image = avg_time / bs
            fps = bs / avg_time

            print(f"  {bs:<10} {avg_time:<12.4f} {per_image:<12.4f} {fps:<12.2f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {bs:<10} 显存不足")
                torch.cuda.empty_cache()
                break

    print("  " + "-" * 50)

    # ==================== 生成对比图表 ====================
    print("\n" + "=" * 80)
    print("【生成可视化结果】")
    print("=" * 80)

    fig = plt.figure(figsize=(18, 10))

    # 1. 推断时间对比
    ax1 = plt.subplot(2, 3, 1)
    platforms = list(results.keys())
    times = [results[p]['avg_time'] for p in platforms]
    stds = [results[p]['std_time'] for p in platforms]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax1.bar(platforms, times, yerr=stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('推断时间 (秒)', fontsize=12, fontweight='bold')
    ax1.set_title('推断时间对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{t:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. 吞吐量对比
    ax2 = plt.subplot(2, 3, 2)
    fps_values = [results[p]['fps'] for p in platforms]
    bars = ax2.bar(platforms, fps_values, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('吞吐量 (FPS)', fontsize=12, fontweight='bold')
    ax2.set_title('处理速度对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, fps in zip(bars, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{fps:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 加速比对比
    ax3 = plt.subplot(2, 3, 3)
    baseline = results[platforms[0]]['avg_time']
    speedups = [baseline / results[p]['avg_time'] for p in platforms]
    bars = ax3.bar(platforms, speedups, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_ylabel('加速比', fontsize=12, fontweight='bold')
    ax3.set_title('相对加速效果', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{speedup:.1f}x', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    # 4-6. 输出图像对比
    images = [output_image_cpu, output_image_quant, output_image_gpu]
    titles = ['CPU (FP32)', 'CPU (INT8)', 'GPU (RTX 4060)']

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(2, 3, 4 + i)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('实时风格迁移完整性能测试报告', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('complete_performance_report.png', dpi=200, bbox_inches='tight')
    print("  ✓ 完整报告已保存: complete_performance_report.png")

    # ==================== 最终总结 ====================
    print("\n" + "=" * 80)
    print("【最终总结】")
    print("=" * 80)

    print(f"\n  {'平台':<20} {'时间(秒)':<12} {'FPS':<10} {'加速比':<10}")
    print("  " + "-" * 55)
    for i, platform in enumerate(platforms):
        stats = results[platform]
        speedup = baseline / stats['avg_time']
        print(f"  {platform:<20} {stats['avg_time']:<12.4f} {stats['fps']:<10.2f} {speedup:<10.2f}x")
    print("  " + "-" * 55)

    gpu_speedup = stats_cpu['avg_time'] / stats_gpu['avg_time']

    print(f"\n  🎯 关键指标:")
    print(f"     GPU相对CPU加速: {gpu_speedup:.1f}x")
    print(f"     GPU吞吐量: {stats_gpu['fps']:.1f} FPS")
    print(f"     量化压缩比: ~4x")
    print(f"     量化加速比: {stats_cpu['avg_time'] / stats_quant['avg_time']:.2f}x")

    if gpu_speedup >= 20:
        print(f"\n  ⭐⭐⭐ 优秀！GPU加速效果非常显著")
    elif gpu_speedup >= 15:
        print(f"\n  ⭐⭐ 良好！GPU加速效果明显")
    elif gpu_speedup >= 10:
        print(f"\n  ⭐ 合格，GPU加速达到预期")
    else:
        print(f"\n  ⚠️ GPU加速效果低于预期")

    if stats_gpu['fps'] >= 30:
        print(f"  ✓ 达到实时处理标准 (>30 FPS)")

    print(f"\n  生成文件:")
    print(f"     - output_cpu.jpg")
    print(f"     - output_quantized.jpg")
    print(f"     - output_gpu.jpg")
    print(f"     - complete_performance_report.png")

    print("\n" + "=" * 80)
    print(" " * 30 + "实验完成！")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()