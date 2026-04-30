import torch
import numpy as np
import cv2
import os
from preprocess import preprocess_image, cv_imread

# 从 model.py 导入统一的模型
from model import UNet

def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """加载模型，带兼容性处理"""
    model = UNet().to(device)
    model.eval()
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"模型加载成功: {model_path}")
        else:
            print("警告：模型文件不存在，使用随机权重")
    except Exception as e:
        print(f"模型加载失败：{e}")
        print("使用随机权重")
    
    return model

def predict_crack(model, img_path: str, threshold: float = 0.3):
    """预测裂纹"""
    device = next(model.parameters()).device
    img_chw = preprocess_image(img_path)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
    
    binary_mask = (prob > threshold).astype(np.uint8) * 255
    return binary_mask

def overlay_mask(img_path: str, mask: np.ndarray, alpha: float = 0.5, color: tuple = (0, 0, 255)):
    """将mask叠加到原图上"""
    img = cv_imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 255] = color
    blended = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
    return blended

def predict_and_overlay(model, img_path: str, threshold: float = 0.3):
    """预测并叠加显示"""
    mask = predict_crack(model, img_path, threshold)
    overlay = overlay_mask(img_path, mask)
    return mask, overlay
