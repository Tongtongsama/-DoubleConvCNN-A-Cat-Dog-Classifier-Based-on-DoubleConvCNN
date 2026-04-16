import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import csv
import argparse
from tqdm import tqdm

# 修改点 1：确保能从外部导入你的模型定义
# 假设你的文件结构是：
# project/
# ├── mymodel.py          (存放 DoubleConvCNN)
# ├── predict_folder.py   (本脚本)
# └── models/
#     └── best_doubleconv_model.pth
from mymodel import DoubleConvCNN 

def predict_folder(folder_path, model_path, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    
    # 修改点 2：增强路径兼容性
    # 如果输入的 model_path 不存在，尝试去 models/ 文件夹下寻找
    if not os.path.exists(model_path):
        alternative_path = os.path.join('models', model_path)
        if os.path.exists(alternative_path):
            model_path = alternative_path
        else:
            print(f"❌ 错误：找不到模型文件 {model_path}")
            return

    # 1. 实例化模型
    model = DoubleConvCNN(num_classes=2)
    
    # 2. 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容两种保存格式：纯权重或带字典的 checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载权重出错: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 3. 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_names = ['cat', 'dog']
    results = []

    # 4. 获取并排序文件
    raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    try:
        image_files = sorted(raw_files, key=lambda x: int(os.path.splitext(x)[0]))
    except:
        image_files = sorted(raw_files)

    # 5. 循环预测
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                results.append([filename, class_names[predicted.item()], f"{confidence.item():.4f}"])
            except Exception as e:
                print(f"图片 {filename} 出错: {e}")

    # 6. 保存结果
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Label', 'Confidence'])
        writer.writerows(results)
    
    print(f"完成！结果保存在: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='图片文件夹路径')
    # 修改点 3：默认路径直接指向 models 文件夹
    parser.add_argument('--model', type=str, default='models/best_doubleconv_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default='predictions.csv')
    
    args = parser.parse_args()
    predict_folder(args.folder_path, args.model, args.output)