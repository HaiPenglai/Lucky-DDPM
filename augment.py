import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# --- 配置 ---
SOURCE_DIR = "data"
TARGET_DIR = "dataset"
TARGET_COUNT = 2000  # 目标总张数
IMG_SIZE = 128

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def get_augmented_image(img):
    """对单张图片进行随机几何增强"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 1. 随机旋转角度 (0-360)
    angle = random.uniform(0, 360)
    
    # 2. 随机缩放 (0.9 - 1.1)
    scale = random.uniform(0.9, 1.1)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行变换，borderValue=(0,0,0) 确保背景纯黑
    augmented = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # 3. 随机翻转 (50% 概率水平，50% 概率垂直)
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1) # 水平
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 0) # 垂直

    return augmented

def main():
    # 获取原始切好的图片列表
    source_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    num_sources = len(source_files)
    
    if num_sources == 0:
        print(f"错误: {SOURCE_DIR} 文件夹中没有找到图片！")
        return

    print(f"检测到 {num_sources} 张原始图片，开始扩充至 {TARGET_COUNT} 张...")

    # 计算每张原图需要生成的变体数量
    variants_per_image = TARGET_COUNT // num_sources
    remainder = TARGET_COUNT % num_sources

    count = 0
    for i, filename in enumerate(tqdm(source_files)):
        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # 每张图至少生成若干个变体
        current_variants = variants_per_image + (1 if i < remainder else 0)
        
        # 第一张通常保留原图（Resize确保尺寸统一）
        cv2.imwrite(os.path.join(TARGET_DIR, f"aug_{count}_orig.jpg"), img)
        count += 1
        
        for j in range(current_variants - 1):
            aug_img = get_augmented_image(img)
            save_name = f"aug_{count}_{j}.jpg"
            cv2.imwrite(os.path.join(TARGET_DIR, save_name), aug_img)
            count += 1

    print(f"\n[处理完成] 已在 {TARGET_DIR} 目录生成 {count} 张增强图片！")

if __name__ == "__main__":
    main()