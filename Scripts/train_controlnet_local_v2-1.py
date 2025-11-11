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

# 基本配置（与 v1 一致）
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-canny-sdxl"
# 默认切到 *_v2.csv
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2.csv"
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sdxl"
device    = torch.device("cuda")

# 解析模式：cf2octa / octa2cf（默认 cf2octa）
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
parser.add_argument("--n", "--name", dest="name", default='v2_cf_input')
parser.add_argument("--train_csv", default=train_csv)
parser.add_argument("--val_csv", default=val_csv)
args, _ = parser.parse_known_args()

# 根据 name 组装最终输出目录（按 v2 单独命名空间）
v2_tag = args.name or 'v2_cf_input'
out_dir = os.path.join(out_root, args.mode, v2_tag)
os.makedirs(out_dir, exist_ok=True)

# 数据
SIZE=512
to_rgb = transforms.Compose([transforms.Resize((SIZE,SIZE)),
                             transforms.ConvertImageDtype(torch.float32)])

def pil_to_tensor_rgb(img):
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE,SIZE])

# v2: 将分割路径映射为原始 CF 图路径的兜底逻辑（去掉目录名中的 `seg_` 前缀）
def _strip_seg_prefix_in_path(path: str) -> str:
    if not path:
        return path
    parts = path.split(os.sep)
    new_parts = []
    for p in parts:
        if p.startswith("seg_"):
            new_parts.append(p.replace("seg_", "", 1))
        else:
            new_parts.append(p)
    return os.sep.join(new_parts)

# v2: 强制 ControlNet 的 cond 为原始域原图（优先 cf_path/octa_path），回退到 seg 推断

def _pick_paths_v2(row):
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt  = row.get("target_path")

    if args.mode == "cf2octa":
        cond_cf = cf or _strip_seg_prefix_in_path(cond) if (cf or cond) else None
        dst_octa = octa or tgt
        if not cond_cf or not dst_octa:
            raise ValueError(f"cf2octa 需要 cf_path/cond_path 与 octa_path/target_path 至少各提供一个。row={row}")
        return cond_cf, dst_octa
    else:
        cond_octa = octa or _strip_seg_prefix_in_path(tgt or cond) if (octa or tgt or cond) else None
        dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
        if not cond_octa or not dst_cf:
            raise ValueError(f"octa2cf 需要 octa_path/target_path/cond_path 与 cf_path/cond_path/target_path 提供可推断原图的路径。row={row}")
        return cond_octa, dst_cf

class PairCSV(Dataset):
    def __init__(self, csv_path):
        self.rows=[]
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: self.rows.append(r)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r=self.rows[idx]
        src_path, dst_path = _pick_paths_v2(r)
        cond = pil_to_tensor_rgb(Image.open(src_path))
        tgt  = pil_to_tensor_rgb(Image.open(dst_path))
        # VAE 输入需 [-1,1]
        tgt = tgt*2-1
        return cond, tgt

train_ds = PairCSV(args.train_csv)
val_ds   = PairCSV(args.val_csv)
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

max_steps = 5000
global_step = 0
pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval(); pipe.text_encoder_2.eval()
pipe.controlnet.train()

# 计时：用于统计每100 step耗时
if device.type == "cuda":
    torch.cuda.synchronize()
t_block = time.time()

print(f"\n[v2] 模型加载完成，开始进入训练阶段... 模式: {args.mode} | 输出: {out_dir} | 训练CSV: {args.train_csv}")

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
            print(f"[v2] step {global_step}/{max_steps} loss {loss.item():.4f} | 100step {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_block = time.time()
        if global_step % 1000 == 0:
            # 保存快照
            pipe.controlnet.save_pretrained(os.path.join(out_dir, f"step_{global_step}"))

# 最终保存
pipe.controlnet.save_pretrained(out_dir)
print("[v2] saved to", out_dir) 