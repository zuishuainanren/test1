import os
import shutil
from PIL import Image
import random

def clean_directory(dir_path):
    """
    清理目录：如果目录存在则删除并重新创建
    """
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"清理目录时出错: {e}")
            return False
    try:
        os.makedirs(dir_path)
        return True
    except Exception as e:
        print(f"创建目录时出错: {e}")
        return False

def create_yolo_dataset(source_dir, output_dir):
    """
    将数据集转换为YOLO格式，只包含水果类别
    """
    # 清理并创建输出目录
    if not clean_directory(output_dir):
        print("无法创建输出目录，请检查权限或是否有程序占用该目录")
        return False
    
    try:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    except Exception as e:
        print(f"创建子目录时出错: {e}")
        return False
    
    # 定义非水果类别
    non_fruit_categories = [
        'Potato', 'Onion', 'Kohlrabi', 'Zucchini', 'Pepper', 'Pepino',
        'Nut', 'Hazelnut', 'Pistachio', 'Walnut', 'Cauliflower', 'Broccoli',
        'Cabbage', 'Mushroom', 'Corn', 'Carrot', 'Cucumber', 'Eggplant',
        'Ginger', 'Tomato', 'Garlic', 'Bean', 'Pea', 'Spinach'
    ]
    
    # 获取所有类别并过滤非水果
    all_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    classes = []
    
    # 过滤非水果类别
    for cls in all_classes:
        is_non_fruit = False
        for non_fruit in non_fruit_categories:
            if non_fruit.lower() in cls.lower():
                is_non_fruit = True
                break
        if not is_non_fruit:
            classes.append(cls)
    
    print(f"找到 {len(classes)} 个水果类别:")
    for cls in classes:
        print(f"- {cls}")
    
    try:
        # 保存类别映射
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
    except Exception as e:
        print(f"写入类别文件时出错: {e}")
        return False
    
    # 处理每个类别
    total_images = 0
    for cls_id, cls_name in enumerate(classes):
        cls_dir = os.path.join(source_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        
        # 获取该类别的所有图片
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        
        print(f"处理类别 {cls_name} ({len(images)} 张图片)")
        
        for img_name in images:
            try:
                # 读取图片获取尺寸
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path)
                width, height = img.size
                
                # 创建YOLO格式的标签
                label = f"{cls_id} 0.5 0.5 0.8 0.8\n"
                
                # 保存图片
                dst_img_path = os.path.join(output_dir, 'images', img_name)
                shutil.copy2(img_path, dst_img_path)
                
                # 保存标签
                label_name = os.path.splitext(img_name)[0] + '.txt'
                with open(os.path.join(output_dir, 'labels', label_name), 'w') as f:
                    f.write(label)
                
            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {e}")
                continue
    
    print(f"\n总共处理了 {total_images} 张水果图片")
    return True

def split_train_val(dataset_dir, train_ratio=0.8):
    """
    将数据集分割为训练集和验证集
    """
    try:
        # 创建训练集和验证集目录
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                os.makedirs(os.path.join(dataset_dir, split, subdir), exist_ok=True)
        
        # 获取所有图片和标签
        images = os.listdir(os.path.join(dataset_dir, 'images'))
        labels = os.listdir(os.path.join(dataset_dir, 'labels'))
        
        # 随机打乱
        random.shuffle(images)
        
        # 分割数据集
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 移动文件
        for img_name in train_images:
            try:
                # 移动图片
                shutil.move(
                    os.path.join(dataset_dir, 'images', img_name),
                    os.path.join(dataset_dir, 'train', 'images', img_name)
                )
                # 移动对应的标签
                label_name = os.path.splitext(img_name)[0] + '.txt'
                if label_name in labels:
                    shutil.move(
                        os.path.join(dataset_dir, 'labels', label_name),
                        os.path.join(dataset_dir, 'train', 'labels', label_name)
                    )
            except Exception as e:
                print(f"移动训练集文件 {img_name} 时出错: {e}")
                continue
        
        for img_name in val_images:
            try:
                # 移动图片
                shutil.move(
                    os.path.join(dataset_dir, 'images', img_name),
                    os.path.join(dataset_dir, 'val', 'images', img_name)
                )
                # 移动对应的标签
                label_name = os.path.splitext(img_name)[0] + '.txt'
                if label_name in labels:
                    shutil.move(
                        os.path.join(dataset_dir, 'labels', label_name),
                        os.path.join(dataset_dir, 'val', 'labels', label_name)
                    )
            except Exception as e:
                print(f"移动验证集文件 {img_name} 时出错: {e}")
                continue
        
        # 删除原始目录
        try:
            os.rmdir(os.path.join(dataset_dir, 'images'))
            os.rmdir(os.path.join(dataset_dir, 'labels'))
        except Exception as e:
            print(f"删除原始目录时出错: {e}")
        
        return True
        
    except Exception as e:
        print(f"分割数据集时出错: {e}")
        return False

if __name__ == "__main__":
    # 设置路径
    source_dir = r'E:\ProjectPycharm\FruitClassifierVision-main\fruits-360-100x100-main\fruits-360-100x100-main\Training'
    output_dir = 'yolo_dataset'
    
    print("开始准备数据集...")
    
    # 创建YOLO格式数据集
    if create_yolo_dataset(source_dir, output_dir):
        print("YOLO格式数据集创建成功")
        
        # 分割训练集和验证集
        if split_train_val(output_dir):
            print("数据集分割完成")
        else:
            print("数据集分割失败")
    else:
        print("YOLO格式数据集创建失败") 