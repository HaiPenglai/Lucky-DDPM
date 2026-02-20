import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

# --- 配置类 ---
class Config:
    dataset_path = "dataset"         # 增强后的数据集路径
    output_dir = "lucky_clover_model" # 模型保存路径
    image_size = 128                # 分辨率 128x128
    
    # 针对 5090 (32GB) 的调优
    train_batch_size = 128          # 128 尺寸下 128 batch 约占 10-15G 显存
    gradient_accumulation_steps = 1 # 显存充足，无需累加
    learning_rate = 2e-4            # 大 Batch 下稍微调高学习率
    
    num_epochs = 200                # 总训练轮数
    lr_warmup_steps = 500
    save_every_epochs = 10          # 每 10 轮保存一次权重并出图预览
    
    mixed_precision = "bf16"        # 5090 必选 bf16，性能好且稳定
    num_workers = 8                 # 数据读取线程数

class CloverDataset(Dataset):
    def __init__(self, root, transform):
        # 过滤非图片文件
        self.images = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        try:
            image = Image.open(self.images[i]).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"读取图片出错: {self.images[i]}, 错误: {e}")
            # 返回一个随机噪声替代，防止训练中断
            return torch.randn(3, 128, 128)

def train():
    config = Config()
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )

    # 定义针对 128x128 优化的 UNet
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        # 128 分辨率下，下采样 5 次到 4x4 是最佳实践
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",      # 128 -> 64
            "DownBlock2D",      # 64 -> 32
            "DownBlock2D",      # 32 -> 16
            "AttnDownBlock2D",  # 16 -> 8 (中层加入注意力机制)
            "DownBlock2D",      # 8 -> 4
        ),
        up_block_types=(
            "UpBlock2D",        # 4 -> 8
            "AttnUpBlock2D",    # 8 -> 16
            "UpBlock2D",        # 16 -> 32
            "UpBlock2D",        # 32 -> 64
            "UpBlock2D",        # 64 -> 128
        ),
    )

    # ！！重要！！针对 5090 关闭梯度检查点，提升训练速度
    # model.enable_gradient_checkpointing() 

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # 归一化到 [-1, 1]
    ])
    
    dataset = CloverDataset(config.dataset_path, preprocess)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # 准备环境
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 创建保存目录
    if accelerator.is_main_process:
        if not os.path.exists("samples"): os.makedirs("samples")
        if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)

    print(f"开始训练... 目标轮数: {config.num_epochs}")

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_images = batch
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]
                
                # 采样随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), 
                    device=clean_images.device
                ).long()
                
                # 向图片添加噪声
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # 预测噪声
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())
        
# --- 定期保存和采样 ---
        if (epoch + 1) % config.save_every_epochs == 0:
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler
                )
                
                # 1. 保存预览图 (由于 .gitignore 有 /samples/，这里很安全)
                print(f"\n正在生成 Epoch {epoch} 预览图...")
                images = pipeline(batch_size=8, num_inference_steps=50).images
                for i, img in enumerate(images):
                    img.save(f"samples/epoch_{epoch+1}_{i}.png")
                
                # 2. 保存历史权重 (带编号)
                checkpoint_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch+1}")
                pipeline.save_pretrained(checkpoint_path)
                
                # 3. 保存最新权重到 latest 文件夹
                latest_path = os.path.join(config.output_dir, "latest")
                pipeline.save_pretrained(latest_path)
                
                print(f"🎉 模型已更新至: {latest_path}")
                print(f"💾 备份权重已保存: {checkpoint_path}")

    print("所有训练任务已完成！")

if __name__ == "__main__":
    train()