import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QProgressBar, QTextEdit, QMessageBox, QScrollArea,
                           QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect
import torch
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

class FruitDetectorYOLO(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水果检测器 (YOLO)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载YOLO模型
        self.model = self.load_model()
        
        # 类别名称
        self.class_names = ['苹果', '香蕉', '橙子', '草莓', '西瓜',
                           '菠萝', '芒果', '猕猴桃', '梨子', '葡萄',
                           '柠檬', '桃子', '樱桃', '蓝莓', '石榴',
                           '火龙果', '椰子', '木瓜', '柿子', '杏子']
        
        self.init_ui()
    
    def load_model(self):
        try:
            # 加载YOLOv5模型
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path='runs/train/exp/weights/best.pt')
            model.conf = 0.25  # 置信度阈值
            model.iou = 0.45   # NMS IOU阈值
            return model
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型时出错: {str(e)}")
            return None
    
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
        
        # 单张图片检测按钮
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
        
        # 文件夹检测按钮
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
        
        image_title = QLabel("检测结果预览")
        image_title.setFont(QFont("Arial", 12, QFont.Bold))
        image_title.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_title)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        image_layout.addWidget(self.image_label)
        
        display_layout.addWidget(image_frame)
        
        # 结果显示区域
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setStyleSheet("QFrame { background-color: white; }")
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("检测结果")
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
    
    def process_single_image(self, image_path):
        try:
            # 读取图像
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 进行检测
            results = self.model(img)
            
            # 获取检测结果
            detections = results.pandas().xyxy[0]
            
            # 在图像上绘制检测框和标签
            img_with_boxes = img.copy()
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                cls = int(det['class'])
                
                # 绘制边界框
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 准备标签文本
                label = f"{self.class_names[cls]}: {conf:.2%}"
                
                # 绘制标签背景
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_with_boxes, (x1, y1-label_h-10), (x1+label_w, y1), (255, 0, 0), -1)
                
                # 绘制标签文本
                cv2.putText(img_with_boxes, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1)
            
            # 转换为QImage并显示
            height, width, channel = img_with_boxes.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_with_boxes.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 调整图像大小以适应显示区域
            scaled_pixmap = pixmap.scaled(self.image_label.size(), 
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # 显示检测结果
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = f"\n[{current_time}] 检测结果:\n"
            result += f"图片路径: {image_path}\n"
            result += "检测到的物体:\n"
            
            for _, det in detections.iterrows():
                cls = int(det['class'])
                conf = det['confidence']
                result += f"- {self.class_names[cls]}: {conf:.2%}\n"
            
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
    window = FruitDetectorYOLO()
    window.show()
    sys.exit(app.exec_()) 