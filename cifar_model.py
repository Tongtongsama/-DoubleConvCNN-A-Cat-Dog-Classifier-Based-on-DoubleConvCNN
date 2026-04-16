import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvCNN(nn.Module):
    def __init__(self, num_classes=10): # 修改为 10 类
        super(DoubleConvCNN, self).__init__()
        
        def vgg_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # 保持原样的 4 层漏斗结构
        self.features = nn.Sequential(
            vgg_block(3, 32),    # 32 -> 16
            vgg_block(32, 64),   # 16 -> 8
            vgg_block(64, 128),  # 8 -> 4
            vgg_block(128, 256), # 4 -> 2
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # 保持 256 输入
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        init_strategies = {
            nn.Conv2d: lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'),
            nn.BatchNorm2d: lambda m: (nn.init.constant_(m.weight, 1), nn.init.constant_(m.bias, 0)),
            nn.Linear: lambda m: (nn.init.normal_(m.weight, 0, 0.01), nn.init.constant_(m.bias, 0))
        }
        for m in self.modules():
            strategy = init_strategies.get(type(m))
            if strategy:
                strategy(m)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x