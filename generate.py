from diffusers import DDPMPipeline, DDIMScheduler
import torch
import os

model_path = "lucky_clover_model"
save_path = "final_results"
if not os.path.exists(save_path): os.makedirs(save_path)

# 加载模型
pipeline = DDPMPipeline.from_pretrained(model_path).to("cuda")

# 重点：换成 DDIM Scheduler 可以在 50 步内生成极高质量的图，演示效果更好
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

print("正在生成幸运草...")
# 一次生成 16 张
results = pipeline(batch_size=16, num_inference_steps=50).images

for i, img in enumerate(results):
    img.save(f"{save_path}/generated_{i}.png")

print(f"生成完毕，请查看 {save_path} 文件夹！")