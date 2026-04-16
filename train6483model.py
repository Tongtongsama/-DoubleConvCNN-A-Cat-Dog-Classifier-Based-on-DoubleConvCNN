import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
# Device selection: Use GPU if available, otherwise fallback to CPU
# 设备选择：GPU 优先，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 核心模型 Core model (Double convolution + pooling structure)
from mymodel import DoubleConvCNN 

# 2. 训练监控与可视化
# Training monitoring and visualization tools

import matplotlib.pyplot as plt  # 以后用来画 Loss 曲线
from tqdm import tqdm            # 进度条
import time                      # 记录每一轮跑了多久

# 3. 结果保存与辅助工具
# Utilities for saving results and plotting

import os                        # 处理模型保存的路径
from utils_english import (
    plot_training_history_english, # 训练完后，一键生成漂亮的准确率图表Plot training curves
    save_checkpoint               # 训练过程中，把最好的模型“存档”Save best model
)
print(f"使用设备: / Device in use: {device}")
# 超参数：专门为你的 3060 6GB 优化
config = {
    'img_size': 224,        # 图像输入尺寸Input image size
    'batch_size': 32,       # 6GB 显存跑 32 Batch size
    'lr': 0.001,            # 学习率Learning rate
    'epochs': 20,           # 总训练轮数Number of epochs
    'data_path': './data'   # 数据存放路径Dataset path

}
#  数据预处理 / Data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
    transforms.RandomRotation(15),           # 随机旋转 15 度
    transforms.ToTensor(),                   # 关键：把图片转成 0-1 的张量
    transforms.Normalize(                    # 标准化：让模型更容易收敛
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 验证集只做基础处理
# Validation set: only basic preprocessing

val_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 加载数据集# 数据加载 / Dataset loading
# 注意：你的文件夹结构应该是 data/train/cat, data/train/dog
train_set = datasets.ImageFolder(root=f"{config['data_path']}/train", transform=train_transform)
val_set = datasets.ImageFolder(root=f"{config['data_path']}/val", transform=val_transform)
# DataLoader：批量加载数据
# DataLoader: batch data feeding
train_loader = DataLoader(
    train_set, 
    batch_size=config['batch_size'], 
    shuffle=True,       # 训练一定要打乱顺序！Shuffle data
    num_workers=4,      # 你的 3060 笔记本 CPU 应该不错，可以开 4 个线程加速加载Multi-thread loading
    pin_memory=True     # 显存优化：加速从内存到显存的数据传输Faster GPU transfer
)

val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
# 实例化我们共同设计的模型
# 模型、损失函数、优化器
# Model, loss function, optimizer
model = DoubleConvCNN(num_classes=2)

# 【关键点】把模型推送到 GPU
model = model.to(device)

# 定义损失函数：分类任务的标配“交叉熵”# 交叉熵损失函数（分类任务标准）
# CrossEntropy Loss for classification
criterion = nn.CrossEntropyLoss()

# 定义优化器：Adam 是目前最省心的选择
# Adam optimizer
optimizer = optim.Adam(
    model.parameters(), 
    lr=config['lr'], 
    weight_decay=1e-4  # L2 正则化，防止模型过拟合L2 regularization
)

# 学习率调度器
# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3
)
def train_one_epoch(model, loader, optimizer, criterion, device):# 单轮训练函数 / Train one epoch
    model.train()  # 切换到训练模式（启用 Dropout 和 BatchNorm）
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 tqdm 包装 loader，设置进度条描述
    pbar = tqdm(loader, desc="Training", unit="batch")
    
    for inputs, labels in pbar:
        # 1. 把数据搬到 3060 显卡上
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 2. 梯度清零（擦掉上次的错题本）
        optimizer.zero_grad()
        
        # 3. 前向传播（模型做题）
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 4. 反向传播（计算错在哪了）
        loss.backward()
        
        # 5. 更新参数（改正错误）
        optimizer.step()
        
        # 统计数据
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 动态更新进度条右侧的数值
        pbar.set_postfix({
            'Loss': f"{running_loss / (pbar.n + 1):.4f}",
            'Acc': f"{100. * correct / total:.2f}%"
        })
    
    return running_loss / len(loader), 100. * correct / total
def validate_one_epoch(model, loader, criterion, device):
    model.eval()  # 切换到评价模式（冻结 Dropout）
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in tqdm(loader, desc="Validating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total
def train_model():
    # 1. 初始化记录容器
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    patience_counter = 0


    print(f"开始训练！目标设备:Start training on device:  {device}")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # --- 学习阶段 ---
        # 调用我们之前写的 train_one_epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # --- 考试阶段 ---
        # 调用我们之前写的 validate_one_epoch
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # 2. 学习率调度：根据验证集损失调整“学习步长”
        scheduler.step(val_loss)
        
        # 3. 记录历史数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 4. 打印本轮总结
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['epochs']} - {epoch_time:.1f}s")
        print(f"  [Train] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  [Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 5. 保存最佳模型 (利用 utils_english.py 中的功能)       
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 准备存档字典
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }
            save_checkpoint(checkpoint, 'best_doubleconv_model.pth')
            print(f"  发现更好的模型，已存档！准确率: New best model saved! Acc:  {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= 7: # 对应你之前 config 里的 patience
            print(f" 连续 7 轮没有进步，触发早停。Early stopping triggered ")
            break

    # 7. 训练结束，画出曲线图
    plot_training_history_english(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc']
    )
    print(f"训练完成最高验证集准确率 Training completed Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()