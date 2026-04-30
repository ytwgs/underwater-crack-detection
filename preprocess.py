import cv2
import numpy as np
from typing import Tuple

def cv_imread(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

def safe_denoise(img: np.ndarray) -> np.ndarray:
   
    # 双边滤波：保边降噪
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 中值滤波：去除椒盐噪声（3x3 较小核保护细节）
    img = cv2.medianBlur(img, ksize=3)
    
    # 高斯滤波：轻微平滑（3x3 小核，sigma=1 避免模糊边缘）
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=1.0)
    
    return img

def enhance_contrast(img: np.ndarray) -> np.ndarray:
  
    result = np.zeros_like(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    for i in range(3):
        result[:, :, i] = clahe.apply(img[:, :, i])
    
    return result

def underwater_color_correction(img: np.ndarray) -> np.ndarray:
   
    if img is None or img.size == 0:
        raise ValueError("输入图像无效")
    
    # 降噪处理
    img = safe_denoise(img)

    # 对比度拉伸（白平衡）
    img_float = img.astype(np.float32)
    for c in range(3):
        min_val = np.percentile(img_float[:, :, c], 2)
        max_val = np.percentile(img_float[:, :, c], 98)
        if max_val > min_val:
            img_float[:, :, c] = (img_float[:, :, c] - min_val) / (max_val - min_val) * 255
        img_float[:, :, c] = np.clip(img_float[:, :, c], 0, 255)
    img = img_float.astype(np.uint8)

    # 自适应对比度增强
    img = enhance_contrast(img)
    
    # 锐化增强细节
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # 亮度微调
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    
    return img

def preprocess_image(img_path: str, img_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    img = cv_imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    
    img_corrected = underwater_color_correction(img)
    img_resized = cv2.resize(img_corrected, img_size)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    return img_chw
