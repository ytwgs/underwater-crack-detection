import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 从 model.py 导入统一的模型
from model import UNet
# 从 preprocess.py 导入颜色校正函数
from preprocess import cv_imread, underwater_color_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH = 4
EPOCHS = 50  
H = 256
W = 256
IMG = "dataset/train/images"
MASK = "dataset/train/labels"

class Data(Dataset):
    def __init__(self, img, mask):
        self.img = img
        self.mask = mask
        # 支持更多图片格式
        self.files = [f for f in os.listdir(img) if f.lower().endswith(("jpg","jpeg","png","bmp"))]
        print(f"找到 {len(self.files)} 张图片")
        
        # 打印前5个文件名用于调试
        if len(self.files) > 0:
            print("示例文件:", self.files[:5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]
        ip = os.path.join(self.img, name)
        
        # 尝试多种标注文件格式
        mp = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = os.path.join(self.mask, base + ext)
            if os.path.exists(candidate):
                mp = candidate
                break
        
        if mp is None:
            print(f"警告：找不到标注文件 for {name}")
            # 返回全黑标注作为占位
            i = np.zeros((H, W, 3), dtype=np.float32)
            m = np.zeros((H, W), dtype=np.float32)
            return torch.from_numpy(i).permute(2,0,1), torch.from_numpy(m).unsqueeze(0)

        # 读取图像
        i = cv_imread(ip)
        if i is None:
            print(f"读取图像失败: {ip}")
            i = np.zeros((H, W, 3), dtype=np.uint8)
        
        # ========== 应用水下颜色校正和降噪 ==========
        i = underwater_color_correction(i)
        
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = cv2.resize(i, (W, H))
        i = torch.from_numpy(i).permute(2,0,1).float() / 255.0

        # 读取标注
        m = cv_imread(mp)
        if m is None:
            print(f"读取标注失败: {mp}")
            m = np.zeros((H, W), dtype=np.uint8)
        
        if len(m.shape) == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m = cv2.resize(m, (W, H))
        m = torch.from_numpy(m).float().unsqueeze(0) / 255.0
        m = (m > 0.5).float()

        return i, m

def main():
    print(f"使用设备: {DEVICE}")
    
    # 检查数据集是否存在
    if not os.path.exists(IMG):
        print(f"错误：图像目录不存在 {IMG}")
        return
    if not os.path.exists(MASK):
        print(f"错误：标注目录不存在 {MASK}")
        return
    
    dataset = Data(IMG, MASK)
    if len(dataset) == 0:
        print("错误：没有找到任何图片")
        return
    
    loader = DataLoader(dataset, BATCH, shuffle=True)
    model = UNet().to(DEVICE)

    # 加载已有模型（如果有）
    start_epoch = 0
    if os.path.exists("unet_crack.pth"):
        model.load_state_dict(torch.load("unet_crack.pth", map_location=DEVICE))
        print("加载已有模型继续训练")
    else:
        print("从头开始训练")

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (x, y) in enumerate(loader):
            # 检查是否有无效数据
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"跳过第 {batch_idx} 批：包含NaN")
                continue
                
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            
            if torch.isnan(loss):
                print(f"跳过第 {batch_idx} 批：Loss为NaN")
                continue
                
            loss.backward()
            opt.step()
            total_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | 无有效批次")
            break

    torch.save(model.state_dict(), "unet_crack.pth")
    print("训练完成！模型已保存")

if __name__ == "__main__":
    main()
