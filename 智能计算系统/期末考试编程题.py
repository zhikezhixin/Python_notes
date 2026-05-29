import torch                                # 导入 PyTorch 核心库，提供张量和自动求导等功能
import torch.nn as nn                       # 导入神经网络模块别名，包含常用的层（Conv、Linear、BatchNorm 等）
import torch.nn.functional as F             # 导入函数式 API（通常用于无状态操作，如 relu、softmax、pad 等）
import torch.optim as optim
import math
# 1. 三层卷积网络（必考）
class ThreeLayerCNN(nn.Module):              # 定义一个三层卷积网络类，继承自 nn.Module（所有 PyTorch 模型的基类）
    def __init__(self, num_classes=10):      # 构造函数，num_classes 指最终输出的类别数（默认 10）
        super().__init__()                   # 调用父类构造函数，初始化内部状态（必写）
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # conv1: 2D 卷积层，输入通道 3（RGB 图像），输出通道 32，卷积核大小 3x3，padding=1 保持宽高不变（same）
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # conv2: 接受上层 32 通道，输出 64 通道，3x3 卷积，同样使用 padding=1
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # conv3: 继续加深，输出 128 通道，3x3 卷积，padding=1
        self.pool = nn.MaxPool2d(2, 2)      # 最大池化层：kernel_size=2, stride=2，将宽高各缩小 2 倍（下采样）
        self.fc = nn.Linear(128*4*4, num_classes)
        # 全连接层：输入尺寸 128*4*4（假设输入图像为 32x32，经三次 2x2 池化后宽高变为 4x4），输出为 num_classes

    def forward(self, x):                    # 前向传播函数，定义数据如何通过网络（必实现）
        x = self.pool(F.relu(self.conv1(x))) # 1) conv1 -> ReLU 激活 -> maxpool，下采样 2 倍
        x = self.pool(F.relu(self.conv2(x))) # 2) conv2 -> ReLU -> maxpool，下采样到原来的 1/4
        x = self.pool(F.relu(self.conv3(x))) # 3) conv3 -> ReLU -> maxpool，下采样到原来的 1/8（32->16->8->4）
        x = torch.flatten(x, 1)              # 展平张量，保持 batch 维（dim=0），从第 1 维开始展开为向量im
        return self.fc(x)                    # 最后通过线性层输出 logits（未经过 softmax，训练时用 CrossEntropyLoss）

# 2. 残差模块
class ResidualBlock(nn.Module):              # 定义残差块（ResNet 风格的基础块）
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()                   # 初始化父类
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        # conv1: 3x3 卷积，stride 可以不为 1（用于下采样），padding=1 保持感受野，bias=False 常配合 BatchNorm 使用
        self.bn1 = nn.BatchNorm2d(out_channels)
        # bn1: 对 conv1 的输出做批归一化，加速收敛并稳定训练
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        # conv2: 第二个 3x3 卷积，stride=1，保持尺寸
        self.bn2 = nn.BatchNorm2d(out_channels)
        # bn2: 对 conv2 的输出做批归一化

        self.shortcut = nn.Sequential()      # 默认为恒等映射（shortcut），当输入输出通道/尺寸一致时直接相加
        if stride != 1 or in_channels != out_channels:
            # 如果需要下采样（stride!=1）或通道数改变，就用 1x1 卷积和 BN 来调整 shortcut 的维度
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
            # 这里 1x1 卷积改变通道数并在需要时下采样，保证与主路径输出形状一致

    def forward(self, x):                    # 残差块的前向传播
        out = F.relu(self.bn1(self.conv1(x))) # 主路径：conv1 -> BN -> ReLU
        out = self.bn2(self.conv2(out))       # 主路径继续：conv2 -> BN （注意：先 BN 再加 residual，再 ReLU）
        return F.relu(out + self.shortcut(x)) # 将主路径输出与 shortcut 相加后做 ReLU（残差连接）

# 3. 模型保存与加载
torch.save(model.state_dict(), 'model.pth')
model = ThreeLayerCNN()
model.load_state_dict(torch.load('model.pth'));
model.eval()

# 4. 完整训练流程
device = torch.device('cpu')
model = ThreeLayerCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad();
        out = model(x)
        loss = criterion(model(x), y)
        loss.backward();
        optimizer.step()

# 5. 测试与验证流程
model.eval()                                # 切换到评估模式（确保 BN/Dropout 在推理模式）
correct = 0
total = 0
with torch.no_grad():                       # 在评估/推理时禁用梯度计算，节省内存并加速
    for x, y in test_loader:                # 遍历测试集
        x, y = x.to(device), y.to(device)  # 同样把数据移动到设备
        out = model(x)	                    # 前向传播得到输出
        pred = out.argmax(1)                # 取最大概率对应的类别索引（按 dim=1）
        total += y.size(0)                  # 累计样本数（batch 大小）
        correct += (pred == y).sum().item() # 统计预测正确的样本数（pred == y 会产生 bool tensor）

accuracy = 100 * correct / total            # 计算准确率（百分比），注意处理 total=0 的情况以防除零

# 6. 自注意力
class SelfAttention(nn.Module):              # 定义一个简单的自注意力层（非多头），用于序列/通道等场景
    def __init__(self, d):
        super().__init__()                   # 初始化父类
        self.q = nn.Linear(d, d)             # 线性变换 Q（query），把输入维度映射到 d（通常 d_k）
        self.k = nn.Linear(d, d)             # 线性变换 K（key）
        self.v = nn.Linear(d, d)             # 线性变换 V（value）
        self.scale = math.sqrt(d)            # 缩放因子：通常是 sqrt(d_k)，用于稳定 softmax 前的数值范围

    def forward(self, x):                    # x 形状通常是 (batch, seq_len, d) 或 (seq_len, d)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        # 分别通过线性层得到 Q, K, V；形状保持一致，便于后续矩阵乘法
        scores = Q @ K.transpose(-2, -1) / self.scale
        # 计算注意力分数：Q * K^T（最后两个维度做矩阵乘），再除以缩放因子
        attn = F.softmax(scores, dim=-1)    # 对最后一个维度做 softmax，以获得注意力权重分布
        return attn @ V                     # 将注意力权重与 V 做加权求和，得到 self-attention 的输出
