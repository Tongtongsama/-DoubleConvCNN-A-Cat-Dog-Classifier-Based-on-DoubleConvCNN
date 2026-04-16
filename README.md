# Image Classification Project (Cat & Dog / CIFAR-10)
# 图像分类项目 (猫狗识别 / CIFAR-10)

**Student ID / 学号:** G2507017C

---

## 📂 Project Structure / 项目结构

- `mymodel.py`: Core model definition (`DoubleConvCNN`) with "Double-Conv-One-Pool" architecture. / 核心模型定义，采用“双卷一池”架构。
- `cifar_model.py`: Adapted model for the CIFAR-10 dataset. / 针对 CIFAR-10 数据集适配的模型变体。
- `train6483model.py`: Training script for the Cat vs. Dog dataset. / 猫狗分类数据集的训练脚本。
- `train_cifar.py`: Training script for the CIFAR-10 10-class task. / CIFAR-10 十分类任务的训练脚本。
- `predicttest.py`: Batch inference script that outputs results to CSV. / 批量预测程序，支持将结果导出为 CSV。
- `utils_english.py`: Utilities for plotting (Loss/Acc) and checkpoint management. / 绘图及模型保存加载等辅助工具。
- `models/`: Directory for trained `.pth` weight files. / 存放训练好的模型权重目录。
- `data/` & `test/`: Folders for datasets and test images. / 用于存放数据集和测试图片的文件夹。

---

## 🚀 Model Advantages / 模型优势

1.  **VGG-style Feature Extraction / VGG风格特征提取**: 
    Uses stacked 3x3 convolutions to increase network depth and non-linearity while maintaining the receptive field. / 采用连续 3x3 卷积堆叠，在保持感受野的同时增加深度和非线性。
    

2.  **Lightweight Design / 轻量化设计**: 
    Replaces heavy Fully Connected layers with **Global Average Pooling (GAP)** to drastically reduce parameters and prevent overfitting. / 引入全局平均池化替代全连接层，大幅减少参数量并防止过拟合。

3.  **Smart Initialization / 智能初始化**: 
    Implements **Kaiming Normal** initialization via dictionary mapping for faster convergence on RTX 3060. / 采用 Kaiming 初始化策略，确保在 RTX 3060 等设备上快速收敛。

4.  **High Scalability / 高扩展性**: 
    Easily switches between binary and multi-class (CIFAR-10) classification tasks. / 能够轻松在二分类和多分类（CIFAR-10）任务间切换。

---

## 🛠️ Usage / 使用说明

### 1. Environment Setup / 环境安装
```bash
pip install -r requirements.txt
end

### 2. Training / 模型训练
To train the Cat & Dog classifier / 训练猫狗分类模型:

```bash
python train6483model.py
end

###3. Batch Prediction / 批量预测
Place images in the test/ folder and run / 将图片放入 test/ 文件夹并运行:

```Bash
python predicttest.py ./test
end
