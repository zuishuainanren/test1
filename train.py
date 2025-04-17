import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from baseModel import FruitCNN
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import freeze_support

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=r'E:\ProjectPycharm\FruitClassifierVision-main\fruits-360-100x100-main\fruits-360-100x100-main\Training',
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=r'E:\ProjectPycharm\FruitClassifierVision-main\fruits-360-100x100-main\fruits-360-100x100-main\Test',
        transform=transform
    )

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
    train_fruit_indices = filter_fruit_classes(train_dataset)
    test_fruit_indices = filter_fruit_classes(test_dataset)

    # 创建只包含水果的数据集
    train_dataset = Subset(train_dataset, train_fruit_indices)
    test_dataset = Subset(test_dataset, test_fruit_indices)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 修改num_workers为0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # 修改num_workers为0

    # 打印数据集信息
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # 初始化模型
    model = FruitCNN(num_classes=len(train_dataset.dataset.classes)).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # 训练参数
    num_epochs = 50
    best_acc = 0.0
    best_loss = float('inf')
    patience = 5  # 早停耐心值
    early_stopping_counter = 0

    # 创建保存模型的目录
    os.makedirs('new_model', exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'classes': train_dataset.dataset.classes
            }, f'new_model/best_model_Loss_{best_loss:.4f}_Acc_{best_acc:.2f}%.pth')
        else:
            early_stopping_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 早停检查
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    print('Training finished!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Best validation loss: {best_loss:.4f}')

if __name__ == '__main__':
    freeze_support()
    main() 