import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QProgressBar, QTextEdit, QMessageBox, QScrollArea,
                           QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset
from PIL import Image, ImageDraw
import numpy as np
from baseModel import FruitCNN
import os
from datetime import datetime

class FruitClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水果分类器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FruitCNN(num_classes=180).to(self.device)
        checkpoint = torch.load("new_model/best_model_Loss_0.0404_Acc_99.06%.pth", 
                              map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集以获取类别信息
        self.dataset = datasets.ImageFolder(
            root=r'E:\ProjectPycharm\FruitClassifierVision-main\fruits-360-100x100-main\fruits-360-100x100-main\Test',
            transform=self.transform
        )
        
        # 定义非水果类别
        self.non_fruit_classes = [
            'Potato White', 'Potato Sweet', 'Potato Red Washed', 'Potato Red',
            'Onion White', 'Onion Red Peeled', 'Onion Red',
            'Kohlrabi',
            'Zucchini', 'Zucchini dark',
            'Pepper Yellow', 'Pepper Red', 'Pepper Orange', 'Pepper Green',
            'Pepino',
            'Nut Pecan', 'Nut Forest', 'Hazelnut', 'Pistachio', 'Walnut'
        ]
        
        # 获取水果类别映射
        self.class_names = self.dataset.classes
        self.fruit_indices = self.filter_fruit_classes()
        
        # 存储当前图像和检测结果
        self.current_image = None
        self.current_detections = []
        
        self.init_ui()
    
    def filter_fruit_classes(self):
        # 找出水果类别的索引
        fruit_indices = []
        for idx, class_name in enumerate(self.class_names):
            if not any(non_fruit in class_name for non_fruit in self.non_fruit_classes):
                fruit_indices.append(idx)
        return fruit_indices
    
    def get_fruit_name(self, class_idx):
        # 获取水果的中文名称
        class_name = self.class_names[class_idx]
        # 这里可以添加更多的类别名称映射
        name_map = {
            'Apple': '苹果',
            'Banana': '香蕉',
            'Orange': '橙子',
            'Strawberry': '草莓',
            'Watermelon': '西瓜',
            'Pineapple': '菠萝',
            'Mango': '芒果',
            'Kiwi': '猕猴桃',
            'Pear': '梨子',
            'Grape': '葡萄',
            'Lemon': '柠檬',
            'Peach': '桃子',
            'Cherry': '樱桃',
            'Blueberry': '蓝莓',
            'Pomegranate': '石榴',
            'Pitahaya': '火龙果',
            'Coconut': '椰子',
            'Papaya': '木瓜',
            'Persimmon': '柿子',
            'Apricot': '杏子',
            'Physalis': '灯笼果',
            'Physalis with Husk': '带壳灯笼果'
        }
        
        # 查找最匹配的名称
        for eng_name, chi_name in name_map.items():
            if eng_name.lower() in class_name.lower():
                return chi_name
        
        return class_name  # 如果没有匹配的中文名称，返回原始英文名称
        
    def init_ui(self):
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧控制面板
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f0f0f0; }")
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # 控制面板标题
        control_title = QLabel("控制面板")
        control_title.setFont(QFont("Arial", 12, QFont.Bold))
        control_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_title)
        
        # 单张图片识别按钮
        self.single_image_btn = QPushButton("选择单张图片")
        self.single_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.single_image_btn.clicked.connect(self.select_single_image)
        control_layout.addWidget(self.single_image_btn)
        
        # 文件夹识别按钮
        self.folder_btn = QPushButton("选择文件夹")
        self.folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.folder_btn.clicked.connect(self.select_folder)
        control_layout.addWidget(self.folder_btn)
        
        # 清空结果按钮
        self.clear_btn = QPushButton("清空结果")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_results)
        control_layout.addWidget(self.clear_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
            }
        """)
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_frame)
        
        # 右侧显示区域
        display_frame = QFrame()
        display_frame.setFrameShape(QFrame.StyledPanel)
        display_layout = QVBoxLayout(display_frame)
        display_layout.setSpacing(10)
        display_layout.setContentsMargins(10, 10, 10, 10)
        
        # 图片显示区域
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_frame.setStyleSheet("QFrame { background-color: white; }")
        image_layout = QVBoxLayout(image_frame)
        
        image_title = QLabel("图片预览")
        image_title.setFont(QFont("Arial", 12, QFont.Bold))
        image_title.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_title)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        image_layout.addWidget(self.image_label)
        
        display_layout.addWidget(image_frame)
        
        # 结果显示区域
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setStyleSheet("QFrame { background-color: white; }")
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("识别结果")
        result_title.setFont(QFont("Arial", 12, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_title)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        result_layout.addWidget(self.result_text)
        
        display_layout.addWidget(result_frame)
        
        main_layout.addWidget(display_frame)
        
    def select_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.process_single_image(file_path)
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.process_folder(folder_path)
    
    def clear_results(self):
        self.result_text.clear()
        self.image_label.clear()
        self.status_label.setText("就绪")
        self.progress_bar.setValue(0)
    
    def draw_detection_boxes(self, image, detections):
        # 创建一个绘图对象
        draw = ImageDraw.Draw(image)
        
        # 定义颜色
        box_color = (255, 0, 0)  # 红色边框
        text_color = (255, 255, 255)  # 白色文字
        
        # 为每个检测结果绘制边框和标签
        for fruit_name, confidence, box in detections:
            # 绘制边框
            draw.rectangle(box, outline=box_color, width=2)
            
            # 准备标签文本
            label = f"{fruit_name}: {confidence:.1%}"
            
            # 绘制标签背景
            text_bbox = draw.textbbox((box[0], box[1]-20), label)
            draw.rectangle((text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2), 
                         fill=(255, 0, 0))
            
            # 绘制标签文本
            draw.text((box[0], box[1]-20), label, fill=text_color)
        
        return image
    
    def process_single_image(self, image_path):
        try:
            # 打开并处理图像
            image = Image.open(image_path)
            # 保持宽高比例
            width, height = image.size
            max_size = 400
            ratio = min(max_size/width, max_size/height)
            new_size = (int(width*ratio), int(height*ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 存储原始图像
            self.current_image = image.copy()
            
            # 进行预测
            with torch.no_grad():
                # 预处理图像
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # 获取预测结果和前5个最可能的类别
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                top5_prob, top5_indices = torch.topk(probabilities[0], 5)
                
                # 创建检测结果列表
                self.current_detections = []
                
                # 对于概率大于阈值的检测结果，添加到列表中
                threshold = 0.2  # 设置阈值
                for prob, idx in zip(top5_prob, top5_indices):
                    if prob.item() > threshold:
                        fruit_name = self.get_fruit_name(idx.item())
                        # 创建一个简单的检测框（这里使用整个图像区域）
                        box = (10, 10, new_size[0]-10, new_size[1]-10)
                        self.current_detections.append((fruit_name, prob.item(), box))
                
                # 在图像上绘制检测框
                image_with_boxes = self.current_image.copy()
                image_with_boxes = self.draw_detection_boxes(image_with_boxes, self.current_detections)
                
                # 转换为QImage并显示
                if image_with_boxes.mode == 'RGB':
                    data = image_with_boxes.tobytes("raw", "RGB")
                    qimage = QImage(data, image_with_boxes.width, image_with_boxes.height, 
                                  image_with_boxes.width * 3, QImage.Format_RGB888)
                else:
                    image_with_boxes = image_with_boxes.convert('RGB')
                    data = image_with_boxes.tobytes("raw", "RGB")
                    qimage = QImage(data, image_with_boxes.width, image_with_boxes.height,
                                  image_with_boxes.width * 3, QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)
                
                # 获取当前时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 显示结果
                result = f"\n[{current_time}] 单张图片识别结果:\n"
                result += f"图片路径: {image_path}\n"
                result += "检测结果：\n"
                
                for fruit_name, confidence, _ in self.current_detections:
                    result += f"- {fruit_name}: {confidence:.2%}\n"
                
                result += "-" * 50 + "\n"
                self.result_text.append(result)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片时出错: {str(e)}")
            
    def process_folder(self, folder_path):
        try:
            # 获取所有图片文件
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件")
                return
            
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.result_text.append(f"\n[{current_time}] 开始批量处理文件夹: {folder_path}\n")
            
            # 处理每张图片
            total = len(image_files)
            for i, file_name in enumerate(image_files):
                file_path = os.path.join(folder_path, file_name)
                
                # 更新进度
                progress = (i + 1) / total * 100
                self.progress_bar.setValue(int(progress))
                self.status_label.setText(f"处理中: {file_name}")
                QApplication.processEvents()
                
                # 处理单张图片
                self.process_single_image(file_path)
            
            self.status_label.setText("处理完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理文件夹时出错: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitClassifierGUI()
    window.show()
    sys.exit(app.exec_()) 