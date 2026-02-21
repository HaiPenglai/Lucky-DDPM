from diffusers import DDPMPipeline, DDIMScheduler
import torch
import os

# model_path 指向你想测试的具体权重文件夹
model_path = "lucky_clover_model/checkpoint-epoch-100" 
base_save_path = "final_results"

# 自动获取模型文件夹名称（例如 checkpoint-epoch-200）
model_name = os.path.basename(os.path.normpath(model_path))
# 构建最终保存路径：final_results/checkpoint-epoch-200
save_path = os.path.join(base_save_path, model_name)

if not os.path.exists(save_path): 
    os.makedirs(save_path)

# 检查路径是否存在
if not os.path.exists(model_path):
    print(f"错误：找不到模型路径 {model_path}")
    exit()

print(f"正在加载模型: {model_path}")
print(f"结果将保存至: {save_path}")

# 训练时用了 bf16，推理时由于显存充足（5090），默认 FP32 即可。
# 如需极致速度，可加入 torch_dtype=torch.bfloat16
pipeline = DDPMPipeline.from_pretrained(model_path).to("cuda")

# 取消下行注释即切换为 DDIM 调度器。
# 实测结论：在本项目中 DDPM 效果远好于 DDIM，DDIM 会引入大量背景噪点。
# pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

print("正在生成幸运草...")

# 虽然模型训练了 1000 步时间步，但通过 Scheduler 跳步采样，50 步即可兼顾速度与质量。
results = pipeline(batch_size=16, num_inference_steps=50).images

for i, img in enumerate(results):
    # 保存文件名包含模型名
    img.save(f"{save_path}/{model_name}_generated_{i}.png")

print(f"生成完毕！请在 {save_path} 文件夹中对比效果。")