'''
使用SDXL+ControlNet - v2配准数据集版本 (v2-2-2)

【版本说明】
就是v2换成配准数据集，其他完全保持一致。
核心改动：在v2基础上增加配准矩阵应用逻辑，支持空间对齐的CF-OCTA配对数据

【与v2的对比】
相同点（完全保持v2的训练配置）：
  * 训练尺寸：768×768
  * ControlNet输入保持[0,1]范围
  * VAE输入使用[-1,1]范围
  * 只使用 Noise Prediction Loss（扩散模型核心）
  * 不使用梯度裁剪
  * 使用显存优化（attention slicing + VAE tiling）

唯一不同（数据处理）：
  * v2：直接resize原始图像到768×768
  * v2-2-2：先应用配准矩阵到原始尺寸，再resize到768×768
  * 配准逻辑：OCTA→CF或CF→OCTA，避免空间错位

【额外功能】（不影响核心训练）
- 平均loss输出（100步均值，更稳定的监控指标）
- 第一个step自动保存调试图像，验证配准效果
- 支持断点续训和optimizer状态恢复

数据集: 配准好的CF、OCTA图像对
训练模式: cf2octa / octa2cf
'''
import os, csv, math, itertools
import torch, numpy as np
import cv2
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
# 默认切到 *_v2-2_repaired.csv（使用修正后命名的配准数据集）
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sdxl"
device    = torch.device("cuda")

# 数据
SIZE=768
to_rgb = transforms.Compose([transforms.Resize((SIZE,SIZE)),
                             transforms.ConvertImageDtype(torch.float32)])

def pil_to_tensor_rgb(img):
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE,SIZE])

def load_affine_matrix(txt_path):
    """加载 2x3 仿射变换矩阵"""
    matrix = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)  # 2x3 矩阵

def apply_affine_registration(img_pil, affine_matrix, output_size=(768, 768)):
    """应用仿射变换配准图像"""
    img_np = np.array(img_pil)
    registered = cv2.warpAffine(img_np, affine_matrix, output_size, 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
    return Image.fromarray(registered)

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
# v2-2: 支持配准数据集，返回配准矩阵路径

def _pick_paths_v2(row):
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt  = row.get("target_path")
    affine_cf_to_octa = row.get("affine_cf_to_octa_path", "")
    affine_octa_to_cf = row.get("affine_octa_to_cf_path", "")

    # 依赖全局 args.mode（在 main() 中赋值）
    if args.mode == "cf2octa":
        # CF 作为条件，OCTA 作为目标（OCTA 需配准到 CF 空间）
        cond_cf = cf or _strip_seg_prefix_in_path(cond) if (cf or cond) else None
        dst_octa = octa or tgt
        if not cond_cf or not dst_octa:
            raise ValueError(f"cf2octa 需要 cf_path/cond_path 与 octa_path/target_path 至少各提供一个。row={row}")
        # 修正：OCTA配准到CF，需要用 OCTA→CF 矩阵
        return cond_cf, dst_octa, affine_octa_to_cf, True
    else:  # octa2cf
        # OCTA 作为条件，CF 作为目标（CF 需配准到 OCTA 空间）
        cond_octa = octa or _strip_seg_prefix_in_path(tgt or cond) if (octa or tgt or cond) else None
        dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
        if not cond_octa or not dst_cf:
            raise ValueError(f"octa2cf 需要 octa_path/target_path/cond_path 与 cf_path/cond_path/target_path 提供可推断原图的路径。row={row}")
        # 修正：CF配准到OCTA，需要用 CF→OCTA 矩阵
        return cond_octa, dst_cf, affine_cf_to_octa, True

class PairCSV(Dataset):
    def __init__(self, csv_path):
        self.rows=[]
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: self.rows.append(r)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        cond_path, tgt_path, affine_path, need_register = _pick_paths_v2(r)
        
        # 加载条件图（不需要配准）
        cond_pil = Image.open(cond_path)
        
        # 加载目标图
        tgt_pil = Image.open(tgt_path)
        
        # 如果需要配准且配准矩阵存在
        # 关键修复：先配准到条件图的原始大小，再统一resize
        if need_register and affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            # 输出大小应该是条件图（CF）的大小，与配准矩阵计算时一致
            cond_size = (cond_pil.width, cond_pil.height)
            tgt_pil = apply_affine_registration(tgt_pil, affine_matrix, cond_size)
        
        # 统一resize到训练尺寸
        cond = pil_to_tensor_rgb(cond_pil)
        tgt = pil_to_tensor_rgb(tgt_pil)
        
        # ControlNet输入保持[0,1]（与v2一致），VAE需要[-1,1]
        # cond保持[0,1]
        tgt = tgt * 2 - 1
        
        # 返回数据和文件路径（用于调试输出）
        return cond, tgt, cond_path, tgt_path

# 文本与 VAE 编码工具依赖全局 pipe/vae_sf（在 main() 内初始化）
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
    latents = pipe.vae.encode(img).latent_dist.sample() * vae_sf
    return latents

def decode_vae(latents): # latents: scaled
    img = pipe.vae.decode(latents / vae_sf).sample  # [-1,1]
    return img


def main():
    # 解析模式：cf2octa / octa2cf（默认 cf2octa）
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
    parser.add_argument("-n", "--name", "-name", dest="name", default='v2-2-2_registered')
    parser.add_argument("--train_csv", default=train_csv)
    parser.add_argument("--val_csv", default=val_csv)
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="从指定checkpoint恢复训练，例如: /path/to/out_dir/step_6500")
    parser.add_argument("--max_steps", type=int, default=8000, help="总训练步数")
    global args
    args, _ = parser.parse_known_args()

    # 根据 name 组装最终输出目录（按 v2 单独命名空间）
    v2_tag = args.name or 'v2-2_cf_input'
    out_dir = os.path.join(out_root, args.mode, v2_tag)
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_ds = PairCSV(args.train_csv)
    val_ds   = PairCSV(args.val_csv)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    # 模型组件
    os.environ["HF_HUB_OFFLINE"]="1"
    global pipe, vae_sf
    
    print("正在加载模型...")
    
    # 判断是从头训练还是恢复训练
    resume_step = 0
    if args.resume_from:
        # 从checkpoint恢复
        print(f"从checkpoint恢复训练: {args.resume_from}")
        
        # 解析step数（从路径中提取，例如 /path/to/step_6500 -> 6500）
        import re
        match = re.search(r'step_(\d+)', args.resume_from)
        if match:
            resume_step = int(match.group(1))
            print(f"✓ 检测到checkpoint step: {resume_step}")
        else:
            print("⚠️ 无法从路径中解析step数，将从step 0开始")
        
        # 加载ControlNet checkpoint
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_dir,
            controlnet=ControlNetModel.from_pretrained(args.resume_from, local_files_only=True),
            local_files_only=True
        ).to(device)
        pipe.enable_attention_slicing("max")
        pipe.vae.enable_tiling()
        print(f"✓ 已从 {args.resume_from} 加载ControlNet权重")
    else:
        # 从预训练模型开始
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_dir,
            controlnet=ControlNetModel.from_pretrained(ctrl_dir, local_files_only=True),
            local_files_only=True
        ).to(device)
        pipe.enable_attention_slicing("max")
        pipe.vae.enable_tiling()
        print("✓ 模型已加载（FP32 精度）")

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
    
    # 尝试恢复optimizer状态
    if args.resume_from:
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            opt.load_state_dict(torch.load(optimizer_path))
            print(f"✓ 已恢复optimizer状态")
        else:
            print(f"⚠️ 未找到optimizer状态文件，将使用新的optimizer")

    max_steps = args.max_steps
    global_step = resume_step
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval(); pipe.text_encoder_2.eval()
    pipe.controlnet.train()
    
    print("✓ 使用 FP32 精度训练 + 显存优化（attention slicing + VAE tiling）")

    # 计时和loss统计：用于统计每100 step的平均loss和耗时
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_block = time.time()
    loss_accumulator = []  # 累积每个step的loss

    print(f"\n[v2-2-2 配准数据集版本] 模型加载完成，开始进入训练阶段...")
    print(f"  版本定位: v2 + 配准数据集（训练逻辑完全一致）")
    print(f"  模式: {args.mode}")
    print(f"  输出目录: {out_dir}")
    print(f"  训练CSV: {args.train_csv}")
    print(f"\n  【数据配准】")
    print(f"  - cf2octa: OCTA先用OCTA→CF矩阵配准到CF空间，再resize到768")
    print(f"  - octa2cf: CF先用CF→OCTA矩阵配准到OCTA空间，再resize到768")
    print(f"  - CF(768×768) OCTA(400×400) → 配准到768 → 统一resize到768（减少插值损失）")
    print(f"\n  【训练配置】（与v2完全一致）")
    print(f"  - Loss: 仅使用 Noise Prediction Loss")
    print(f"  - 训练尺寸: 768×768")
    print(f"  - 显存优化: attention slicing + VAE tiling")
    print(f"  - 优化器: AdamW (lr=5e-5, weight_decay=1e-2)")
    print(f"\n  【训练进度】")
    if args.resume_from:
        print(f"  - 从 step {resume_step} 恢复训练")
        print(f"  - 目标: step {max_steps}")
        print(f"  - 剩余训练步数: {max_steps - resume_step}")
    else:
        print(f"  - 从头开始训练")
        print(f"  - 总训练步数: {max_steps}")
    print()

    while global_step < max_steps:
        for batch_data in train_loader:
            if global_step >= max_steps: break
            
            # 解包数据（包含路径信息）
            cond, tgt, cond_paths, tgt_paths = batch_data
            cond = cond.to(device)
            tgt  = tgt.to(device)
            b = tgt.shape[0]
            
            # 第一个step保存调试图像
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 提取文件名（去掉路径和扩展名）
                cf_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                octa_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                # 保存CF条件图（使用原始文件名）
                # cond是[0,1]范围，直接转换
                cond_save = (cond[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_save).save(os.path.join(debug_dir, f"{cf_filename}.png"))
                
                # 保存OCTA目标图（原始文件名_registered）
                # tgt是[-1,1]范围，需要转换到[0,255]
                tgt_save = (tgt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255
                tgt_save = tgt_save.astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{octa_filename}_registered.png"))
                
                print(f"\n✓ 调试图像已保存到: {debug_dir}")
                print(f"  - {cf_filename}.png: 输入ControlNet的CF原图")
                print(f"  - {octa_filename}_registered.png: 配准后的OCTA目标图\n")

            # 准备训练数据（FP32精度）
            with torch.no_grad():
                latents = encode_vae(tgt)  # [b,4,H/8,W/8]
                noise   = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prompt_embeds, pooled_prompt_embeds = get_prompt_embeds(b)
                cond_img = cond
                # 为 SDXL 组装 time_ids
                time_ids = torch.tensor([SIZE, SIZE, 0, 0, SIZE, SIZE], device=device, dtype=prompt_embeds.dtype).unsqueeze(0).repeat(b, 1)

            # 前向传播
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

            # 计算Noise Loss（扩散模型核心损失）
            loss = mse(noise_pred, noise)

            # 反向传播（与v2一致，无梯度裁剪）
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # 累积loss（用于计算100步平均）
            loss_accumulator.append(loss.item())

            global_step += 1
            if global_step % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t_block
                
                # 计算这100步的平均loss
                avg_loss = np.mean(loss_accumulator)
                loss_accumulator = []  # 清空，准备下一个100步
                
                t_val = timesteps[0].item()
                msg = (f"[v2-2-2] step {global_step}/{max_steps} | "
                       f"avg_loss: {avg_loss:.4f} (last_t={t_val:3d}) | "
                       f"100step: {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
                print(msg)
                # 保存日志到对应 step 目录下的 log.txt
                step_log_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_log_dir, exist_ok=True)
                with open(os.path.join(step_log_dir, "log.txt"), "a") as _f:
                    _f.write(msg + "\n")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_block = time.time()
            if global_step % 1000 == 0:
                # 保存快照
                step_save_dir = os.path.join(out_dir, f"step_{global_step}")
                pipe.controlnet.save_pretrained(step_save_dir)
                # 同时保存optimizer状态，方便恢复训练
                torch.save(opt.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                print(f"✓ Checkpoint已保存: {step_save_dir}")

    # 最终保存
    pipe.controlnet.save_pretrained(out_dir)
    torch.save(opt.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    print(f"\n[v2-2-2] 训练完成！ControlNet已保存至: {out_dir}")
    print(f"  版本: v2 + 配准数据集（768×768训练尺寸）")
    print(f"  最终步数: {max_steps}")
    if args.resume_from:
        print(f"  从 step {resume_step} 恢复，训练了 {max_steps - resume_step} 步")
    else:
        print(f"  从头训练了 {max_steps} 步")
    print(f"  使用配准数据可有效减少空间错位，提升模型性能") 


if __name__ == "__main__":
    main()
