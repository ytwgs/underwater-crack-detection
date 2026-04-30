import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox, 
                             QSlider, QSpinBox, QCheckBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from model import UNet

def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def put_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水下裂纹检测系统")
        self.setFixedSize(1600, 950)
        
        self.threshold = 0.5
        self.min_area = 5
        self.max_area = 10000
        self.min_aspect_ratio = 1.5
        self.use_morphology = True
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UNet().to(self.device)
        if os.path.exists("unet_crack.pth"):
            self.model.load_state_dict(torch.load("unet_crack.pth", map_location=self.device))
            print(f"模型加载成功，设备：{self.device}")
        else:
            print("警告：未找到模型文件")
        self.model.eval()
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        title_label = QLabel("水下裂纹智能检测系统")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size:20px; font-weight:bold; margin:10px;")
        left_layout.addWidget(title_label)
        
        img_layout = QHBoxLayout()
        
        self.input_label = QLabel("输入图像")
        self.input_label.setStyleSheet("border:2px solid gray; min-height:500px; background-color:#f0f0f0;")
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.mask_label = QLabel("原始预测掩码")
        self.mask_label.setStyleSheet("border:2px solid gray; min-height:500px; background-color:#f0f0f0;")
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.result_label = QLabel("识别结果")
        self.result_label.setStyleSheet("border:2px solid gray; min-height:500px; background-color:#f0f0f0;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        img_layout.addWidget(self.input_label)
        img_layout.addWidget(self.mask_label)
        img_layout.addWidget(self.result_label)
        left_layout.addLayout(img_layout)
        
        right_widget = QWidget()
        right_widget.setFixedWidth(300)
        right_layout = QVBoxLayout(right_widget)
        
        threshold_group = QGroupBox("检测参数")
        threshold_layout = QVBoxLayout()
        
        threshold_layout.addWidget(QLabel("预测阈值："))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_label = QLabel(f"当前值：{self.threshold:.2f}")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        
        threshold_layout.addWidget(QLabel("最小裂纹面积："))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 500)
        self.min_area_spin.setValue(5)
        self.min_area_spin.valueChanged.connect(lambda v: setattr(self, 'min_area', v))
        threshold_layout.addWidget(self.min_area_spin)
        
        threshold_layout.addWidget(QLabel("最大裂纹面积："))
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(100, 50000)
        self.max_area_spin.setValue(10000)
        self.max_area_spin.valueChanged.connect(lambda v: setattr(self, 'max_area', v))
        threshold_layout.addWidget(self.max_area_spin)
        
        threshold_layout.addWidget(QLabel("最小长宽比："))
        self.aspect_spin = QSpinBox()
        self.aspect_spin.setRange(10, 100)
        self.aspect_spin.setValue(15)
        self.aspect_spin.valueChanged.connect(lambda v: setattr(self, 'min_aspect_ratio', v / 10))
        threshold_layout.addWidget(self.aspect_spin)
        
        threshold_group.setLayout(threshold_layout)
        right_layout.addWidget(threshold_group)
        
        process_group = QGroupBox("处理选项")
        process_layout = QVBoxLayout()
        
        self.morphology_check = QCheckBox("启用形态学处理")
        self.morphology_check.setChecked(True)
        self.morphology_check.stateChanged.connect(lambda v: setattr(self, 'use_morphology', v == Qt.CheckState.Checked.value))
        process_layout.addWidget(self.morphology_check)
        
        process_group.setLayout(process_layout)
        right_layout.addWidget(process_group)
        
        self.select_btn = QPushButton("选择图像")
        self.select_btn.setStyleSheet("background-color:#66b3ff; color:white; padding:10px; font-size:16px; border-radius:5px;")
        self.select_btn.clicked.connect(self.select_image)
        right_layout.addWidget(self.select_btn)
        
        self.reset_btn = QPushButton("重置参数")
        self.reset_btn.setStyleSheet("background-color:#ff9966; color:white; padding:10px; font-size:14px; border-radius:5px;")
        self.reset_btn.clicked.connect(self.reset_parameters)
        right_layout.addWidget(self.reset_btn)
        
        right_layout.addStretch()
        
        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(right_widget, 1)
        
        self.current_image_path = None
        self.current_original_img = None

    def on_threshold_changed(self, value):
        self.threshold = value / 100.0
        self.threshold_label.setText(f"当前值：{self.threshold:.2f}")
        if self.current_original_img is not None:
            self.process_image()

    def reset_parameters(self):
        self.threshold_slider.setValue(50)
        self.min_area_spin.setValue(5)
        self.max_area_spin.setValue(10000)
        self.aspect_spin.setValue(15)
        self.morphology_check.setChecked(True)
        if self.current_original_img is not None:
            self.process_image()

    def apply_morphology(self, mask):
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        return mask

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        self.current_image_path = path
        try:
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(self.input_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.input_label.setPixmap(scaled)
            self.current_original_img = cv_imread(path)
            self.process_image()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    def process_image(self):
        if self.current_original_img is None:
            return
        try:
            h_ori, w_ori = self.current_original_img.shape[:2]
            from preprocess import preprocess_image
            img_np = preprocess_image(self.current_image_path)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(img_tensor)

            pred = torch.sigmoid(pred)
            pred = (pred > self.threshold).float().squeeze().cpu().numpy()
            mask = cv2.resize(pred, (w_ori, h_ori))
            mask = (mask * 255).astype(np.uint8)

            mask_viz = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB)
            h, w, c = mask_viz.shape
            qimg_mask = QImage(mask_viz.data, w, h, c*w, QImage.Format.Format_RGB888)
            self.mask_label.setPixmap(QPixmap.fromImage(qimg_mask).scaled(self.mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            if self.use_morphology:
                mask = self.apply_morphology(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res = self.current_original_img.copy()
            if len(contours) > 0:
                cv2.drawContours(res, contours, -1, (0,0,255), 1)

            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            h, w, c = res_rgb.shape
            qimg = QImage(res_rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
            self.result_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.result_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.statusBar().showMessage(f"完成，检测到 {len(contours)} 处裂纹")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败：{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
