import torch
from ultralytics import YOLO

def train_yolo():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用 YOLOv8 nano 模型
    
    # 开始训练
    results = model.train(
        data='fruit_detection.yaml',      # 数据集配置文件
        epochs=100,                       # 训练轮数
        imgsz=640,                       # 图像大小
        batch=16,                        # 批次大小
        device='cuda' if torch.cuda.is_available() else 'cpu',  # 使用GPU或CPU
        workers=4,                        # 数据加载线程数
        patience=20,                      # 早停轮数
        save=True,                        # 保存模型
        project='runs/train',            # 保存目录
        name='fruit_detection',          # 实验名称
        exist_ok=True,                   # 覆盖已存在的实验目录
        pretrained=True,                 # 使用预训练权重
        optimizer='auto',                # 优化器
        verbose=True,                    # 显示详细训练信息
        seed=42                          # 随机种子
    )
    
    # 打印训练结果
    print("\n训练完成！")
    print(f"最佳mAP: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print(f"最佳精确度: {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"最佳召回率: {results.results_dict['metrics/recall(B)']:.3f}")

if __name__ == '__main__':
    train_yolo() 