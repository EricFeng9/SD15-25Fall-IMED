# /data/student/Fengjunming/SDXL_ControlNet/Scripts/train_controlnet_local.py
import os, csv, math, itertools
import torch, numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionXLControlNetPipeline,
                       ControlNetModel)
import time
import argparse

# 基本配置
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-canny-sdxl"
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs.csv"
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sdxl"
device    = torch.device("cuda")

# 解析模式：cf2octa / octa2cf（默认 cf2octa）
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
parser.add_argument("--n", "--name", dest="name", default='default')
args, _ = parser.parse_known_args()

# 根据 name 组装最终输出目录（name 未提供则保持旧行为）
out_dir = os.path.join(out_root, args.mode, args.name) if args.name else out_root
os.makedirs(out_dir, exist_ok=True)

# 数据
SIZE=768
to_rgb = transforms.Compose([transforms.Resize((SIZE,SIZE)),
                             transforms.ConvertImageDtype(torch.float32)])
def pil_to_tensor_rgb(img):
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE,SIZE])

def _pick_paths(row):
    # 兼容列名：优先 cf_path/oct a_path，其次 cond_path/target_path
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt  = row.get("target_path")
    if cf and octa:
        if args.mode == "cf2octa":
            return cf, octa
        else:
            return octa, cf
    else:
        # 回退到通用列名
        if args.mode == "cf2octa":
            return cond, tgt
        else:
            return tgt, cond

class PairCSV(Dataset):
    def __init__(self, csv_path):
        self.rows=[]
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: self.rows.append(r)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r=self.rows[idx]
        src_path, dst_path = _pick_paths(r)
        cond = pil_to_tensor_rgb(Image.open(src_path))
        tgt  = pil_to_tensor_rgb(Image.open(dst_path))
        # VAE 输入需 [-1,1]
        tgt = tgt*2-1
        return cond, tgt

train_ds = PairCSV(train_csv)
val_ds   = PairCSV(val_csv)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

# 模型组件
os.environ["HF_HUB_OFFLINE"]="1"
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_dir,
    controlnet=ControlNetModel.from_pretrained(ctrl_dir, local_files_only=True),
    local_files_only=True
).to(device)
pipe.enable_attention_slicing("max")
pipe.vae.enable_tiling()

# 冻结 SDXL 主干，只训 ControlNet
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.text_encoder_2.requires_grad_(False)
pipe.controlnet.requires_grad_(True)

noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
opt = torch.optim.AdamW(pipe.controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
mse = nn.MSELoss()
vae_sf = pipe.vae.config.scaling_factor

# 文本嵌入（空prompt）
def get_prompt_embeds(bs):
    prompts = [""]*bs
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    return prompt_embeds, pooled_prompt_embeds

def encode_vae(img): # img: [-1,1]
    latents = pipe.vae.encode(img).latent_dist.sample()*vae_sf
    return latents

max_steps = 2000
global_step = 0
pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval(); pipe.text_encoder_2.eval()
pipe.controlnet.train()

# 计时：用于统计每100 step耗时
if device.type == "cuda":
    torch.cuda.synchronize()
t_block = time.time()

print(f"\n模型加载完成，开始进入训练阶段... 模式: {args.mode}")

while global_step < max_steps:
    for cond, tgt in train_loader:
        if global_step >= max_steps: break
        cond = cond.to(device)
        tgt  = tgt.to(device)
        b = tgt.shape[0]

        with torch.no_grad():
            latents = encode_vae(tgt)  # [b,4,H/8,W/8]
            noise   = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds, pooled_prompt_embeds = get_prompt_embeds(b)
            cond_img = cond
            # 为 SDXL 组装 time_ids（original_size, crop, target_size）
            time_ids = torch.tensor([SIZE, SIZE, 0, 0, SIZE, SIZE], device=device, dtype=prompt_embeds.dtype).unsqueeze(0).repeat(b, 1)

        # 控制分支前向
        down_samples, mid_sample = pipe.controlnet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            controlnet_cond=cond_img, added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}, return_dict=False
        )

        # UNet 预测噪声
        noise_pred = pipe.unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
        ).sample

        loss = mse(noise_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        global_step += 1
        if global_step % 100 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t_block
            print(f"step {global_step}/{max_steps} loss {loss.item():.4f} | 100step {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_block = time.time()
        if global_step % 1000 == 0:
            # 保存快照
            pipe.controlnet.save_pretrained(os.path.join(out_dir, f"step_{global_step}"))

# 最终保存
pipe.controlnet.save_pretrained(out_dir)
print("saved to", out_dir)
