import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ======================
# 1. 设置路径
# ======================
data_dir = r"D:\Zeker\Documents\模式识别于机器学习课程设计\昆虫分类数据集\datasets\datasets\hymenoptera_data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# ======================
# 2. 数据预处理
# ======================
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ======================
# 3. 加载数据
# ======================
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ======================
# 4. 定义模型（迁移学习 ResNet18）
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2分类：蚂蚁 / 蜜蜂

model = model.to(device)

# ======================
# 5. 定义损失函数和优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ======================
# 6. 训练模型
# ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return model

model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# ======================
# 7. 保存模型
# ======================
torch.save(model.state_dict(), "hymenoptera_classifier.pth")
print("模型已保存：hymenoptera_classifier.pth")

import torch
from torchvision import models

# 定义模型结构
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 二分类

# 加载训练好的权重
model.load_state_dict(torch.load(r"D:\CodeManager\PycharmProejcts\Python_notes\hymenoptera_classifier.pth"))
model.eval()

print(model)  # 打印模型结构
