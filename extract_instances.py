import cv2
import numpy as np
import os
import re

# --- 配置 ---
INPUT_DIR = "raw"
OUTPUT_DIR = "data"
TARGET_SIZE = 128
PADDING_RATIO = 0.2 

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def process_image(image_path):
    filename = os.path.basename(image_path)
    prefix = re.match(r"(\d+)", filename).group(1) if re.match(r"(\d+)", filename) else "img"

    img = cv2.imread(image_path)
    if img is None: return
    
    # 1. 基础绿色过滤
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 稍微放宽一点点 V (亮度) 的下限，防止叶片暗部被剔除
    mask_raw = cv2.inRange(hsv, np.array([25, 70, 30]), np.array([90, 255, 255]))
    
    # --- 策略 1：定位掩码（厚掩码） ---
    # 依然使用大核，确保 5 号草（风车草）的散叶子能被框在一起
    kernel_glue = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask_detect = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_glue)
    
    # 2. 使用“定位掩码”找幸运草的大轮廓
    contours, _ = cv2.findContours(mask_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area_ratio = cv2.contourArea(cnt) / (img.shape[0] * img.shape[1])
        if 0.001 < area_ratio < 0.2:
            boxes.append(cv2.boundingRect(cnt))
    
    if not boxes: return

    # 排序
    avg_h = sum([b[3] for b in boxes]) / len(boxes)
    boxes.sort(key=lambda b: (int(b[1] / (avg_h * 0.8)), b[0]))

    for i, (x, y, w, h) in enumerate(boxes):
        # 3. 切出局部图像
        crop_img = img[y:y+h, x:x+w]
        crop_mask_raw = mask_raw[y:y+h, x:x+w]

        # --- 策略 2：精细填充掩码（核心修改点） ---
        # A. 先用极小的核去掉边缘细碎的毛刺
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined_mask = cv2.morphologyEx(crop_mask_raw, cv2.MORPH_OPEN, kernel_tiny)
        
        # B. 找到局部区域内的所有轮廓并【填充】它们
        # 这一步会把叶片中间所有的黑点（孔洞）强制填成白色
        temp_mask = np.zeros(refined_mask.shape, dtype=np.uint8)
        inner_cnts, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in inner_cnts:
            # 只填充面积不是特别小的碎点，进一步过滤背景杂质
            if cv2.contourArea(c) > 50:
                cv2.drawContours(temp_mask, [c], -1, 255, -1) # -1 表示填充内部
        
        # 4. 应用这个“无孔”掩码
        final_crop = cv2.bitwise_and(crop_img, crop_img, mask=temp_mask)

        # 5. 构建画布并居中
        max_dim = int(max(w, h) * (1 + PADDING_RATIO))
        square_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        off_y, off_x = (max_dim - h) // 2, (max_dim - w) // 2
        square_canvas[off_y:off_y+h, off_x:off_x+w] = final_crop
        
        # 6. 缩放到 512x512
        final_img = cv2.resize(square_canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}-{i+1}.jpg"), final_img)

if __name__ == "__main__":
    for f in [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.png'))]:
        print(f"正在处理: {f}...")
        process_image(os.path.join(INPUT_DIR, f))
    print("\n实例提取完成！")