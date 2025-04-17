# %%
# 加载模型进行预测,使用优化后的模型进行预测
import torch
import torch.nn as nn
import torchvision 
from baseModel import FruitCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
batch_size=32

# 定义非水果类别（需要排除的类别）
non_fruit_classes = [
    'Potato White', 'Potato Sweet', 'Potato Red Washed', 'Potato Red',
    'Onion White', 'Onion Red Peeled', 'Onion Red',
    'Kohlrabi',
    'Zucchini', 'Zucchini dark',
    'Pepper Yellow', 'Pepper Red', 'Pepper Orange', 'Pepper Green',
    'Pepino',
    'Nut Pecan', 'Nut Forest', 'Hazelnut', 'Pistachio', 'Walnut'
]

# %%
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 加载预测数据集
predict_dataset = datasets.ImageFolder(
    root=r'E:\ProjectPycharm\FruitClassifierVision-main\fruits-360-100x100-main\fruits-360-100x100-main\Test',
    transform=transform,
)
print(f"Dataset classes: {predict_dataset.classes}")
print(f"Number of classes: {len(predict_dataset.classes)}")
print(f"Number of samples: {len(predict_dataset)}")

# 过滤出水果类别
def filter_fruit_classes(dataset):
    # 获取所有类别
    all_classes = dataset.classes
    # 找出水果类别的索引
    fruit_indices = []
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_name = all_classes[class_idx]
        if not any(non_fruit in class_name for non_fruit in non_fruit_classes):
            fruit_indices.append(idx)
    return fruit_indices

# 获取水果类别的索引
predict_fruit_indices = filter_fruit_classes(predict_dataset)

# 创建只包含水果的数据集
predict_dataset = Subset(predict_dataset, predict_fruit_indices)

predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=True) 


# %%
# 加载保存的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Predict Using",device)
# 加载模型
model = FruitCNN(num_classes=180).to(device)
# 加载模型参数
checkpoint = torch.load(r"new_model/best_model_Loss_0.0404_Acc_99.06%.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["state_dict"]) 

print("\nModel Information:")
print(f"Model class count: {model.classifier[-1].out_features}")
print(f"Model state dict keys: {checkpoint['state_dict'].keys()}")

# 检查第一个样本
sample_image, sample_label = next(iter(predict_loader))
print("\nSample Information:")
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label[0].item()}")
print(f"Sample class name: {predict_dataset.dataset.classes[sample_label[0].item()]}")

# 进行预测测试
with torch.no_grad():
    sample_output = model(sample_image[0:1].to(device))
    sample_prob = torch.softmax(sample_output, dim=1)
    print(f"\nSample prediction probabilities (top 5):")
    top5_probs, top5_indices = torch.topk(sample_prob[0], 5)
    for prob, idx in zip(top5_probs, top5_indices):
        print(f"Class {idx.item()}: {prob.item():.4f}")

model.eval()


# %%
import matplotlib.pyplot as plt
import numpy as np
# 4. 预测和可视化函数
def visualize_predictions(test_loader, model, device, num_samples=32):
    # 获取一个批次的数据
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)  # 获取概率
        _, preds = torch.max(outputs, 1)
    
    # 转换图像为可显示格式
    images = images.cpu().numpy()
    images = images.transpose(0, 2, 3, 1)  # 从 (B, C, H, W) 转为 (B, H, W, C)
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean  # 反归一化
    images = np.clip(images, 0, 1)  # 限制像素值在0-1之间
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(8, 4, i+1)
        plt.imshow(images[i])
        true_label = predict_dataset.dataset.classes[labels[i].item()]
        pred_idx = preds[i].item()
        pred_label = predict_dataset.dataset.classes[pred_idx]
        prob = probabilities[i][pred_idx].item()
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}\nProb: {prob:.2f}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 打印一些统计信息
    correct = (preds == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    print(f"Batch Accuracy: {accuracy:.2%}")
    print(f"Predictions: {preds.tolist()}")
    print(f"Labels: {labels.tolist()}")

# 5. 执行可视化
visualize_predictions(predict_loader, model, device)


