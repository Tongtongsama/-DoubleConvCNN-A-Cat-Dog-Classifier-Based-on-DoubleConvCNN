import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time
import os

# 导入重命名后的模型
from cifar_model import DoubleConvCNN 

# 假设你的 utils_english.py 还在
try:
    from utils_english import plot_training_history_english, save_checkpoint
except ImportError:
    print("Warning: utils_english.py not found. Plotting and checkpoint saving might fail.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'img_size': 32,         # CIFAR10 是 32
    'batch_size': 32,       # 保持你原来的 32
    'lr': 0.001,
    'epochs': 20,
    'data_path': './data' 
}

# 预处理：保持逻辑一致，仅修改 Resize 尺寸
train_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

val_transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 自动下载 CIFAR10
train_set = datasets.CIFAR10(root=config['data_path'], train=True, download=True, transform=train_transform)
val_set = datasets.CIFAR10(root=config['data_path'], train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# 实例化模型
model = DoubleConvCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# --- 复用你原本的训练/验证逻辑函数 ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f"{running_loss / (pbar.n + 1):.4f}", 'Acc': f"{100. * correct / total:.2f}%"})
    return running_loss / len(loader), 100. * correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step(v_loss)
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - {time.time()-start_time:.1f}s | Val Acc: {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            checkpoint = {'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc}
            torch.save(checkpoint, 'best_cifar10_model.pth')

    if 'plot_training_history_english' in globals():
        plot_training_history_english(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

if __name__ == "__main__":
    train_model()