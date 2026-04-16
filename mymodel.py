import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DoubleConvCNN, self).__init__()
        
        # 封装“双卷一池”块        
        # VGG-style double convolution block

        def vgg_block(in_ch, out_ch):
            return nn.Sequential(
                # 第一卷
                # First convolution layer

                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # 第二卷 (保持通道数不变)
                # Second convolution (same number of channels)
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # 一池
                # Max pooling (downsampling)
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # 搭建漏斗结构
        # Feature extractor (funnel structure)
        self.features = nn.Sequential(
            vgg_block(3, 32),    # Block 1: 224 -> 112
            vgg_block(32, 64),   # Block 2: 112 -> 56
            vgg_block(64, 128),  # Block 3: 56 -> 28
            vgg_block(128, 256), # Block 4: 28 -> 14
        )
        
        # Global Average Pooling (reduces 14x14 → 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器   # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 权重初始化策略（字典映射方式）
        # Weight initialization using dictionary mapping
        init_strategies = {
            nn.Conv2d: lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'),
            nn.BatchNorm2d: lambda m: (nn.init.constant_(m.weight, 1), nn.init.constant_(m.bias, 0)),
            nn.Linear: lambda m: (nn.init.normal_(m.weight, 0, 0.01), nn.init.constant_(m.bias, 0))
        }

        for m in self.modules():
            strategy = init_strategies.get(type(m))
            if strategy:
                strategy(m)

    

    # 前向传播
    # Forward pass

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x
    
     
# --- 测试我们的新模型 ---
model = DoubleConvCNN()
x = torch.randn(1, 3, 224, 224)
output = model(x)

print(f"我们的双卷一池模型输出形状 Output shape: {output.shape}")
print(f"模型总参数量:Total parameters: {sum(p.numel() for p in model.parameters()):,}")