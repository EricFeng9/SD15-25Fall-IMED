# -*- coding: utf-8 -*-
"""
CF 生成模型推理脚本 (V22-CF-GEN-TEST)
--------------------------------------

用途：
- 作为 `Scripts_v2/v22/train_cf_gen.py` 训练好的 CF 生成模型的推理脚本。
- 从纯噪声采样生成指定数量的 CF 图像（同域、同风格），用于后续与 cf2fa 串联扩增 CF-FA 成对数据。

使用方式（示例）：
CUDA_VISIBLE_DEVICES=0 python Scripts_v2/v22/test_cf_gen.py \\
  -n 260220_2_no_hf \\
  --amount 1000 \\
  --savedir 1

- 默认：从 `results/out_ctrl_sd15_dual_cf_gen/[name]/best_checkpoint/unet_lora` 读取 LoRA
- 可选：通过 `--ckpt_dir` 指定任意 checkpoint 路径：
    - 传入到 `step_xxxxxx` 目录：脚本会自动拼接 `unet_lora`
    - 或直接传入 `.../unet_lora` 目录

输出目录结构：
results/out_preds_sd15_dual_cf_gen/[name]/[savedir]/
  1/cf.png
  2/cf.png
  ...
  [amount]/cf.png
"""

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
from torchvision import transforms
from vessle_detector import extract_vessel_map

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============ 全局配置 ============

SIZE = 512
DEVICE = torch.device("cuda")
# 训练时使用的基础 SD1.5 模型路径（与 train_cf_gen.py 一致）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
# 训练输出（LoRA checkpoint）根目录（与 train_cf_gen.py 一致）
TRAIN_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual_cf_gen"
# 推理输出根目录
PRED_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual_cf_gen"


# ============ 辅助函数 ============

def get_cf_prompt_embeds(bs, tokenizer, text_encoder):
    """
    与 train_cf_gen.py 对齐：CF 使用固定的彩色眼底 prompt。
    """
    prompt = "color fundus photography, retinal image, medical photography"
    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    将 VAE 解码后的 [-1,1] tensor 转成 PIL.Image
    """
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


@torch.no_grad()
def compute_vessel_ratio_from_tensor(img_tensor_01: torch.Tensor) -> float:
    """
    使用与训练时相同的 extract_vessel_map，对 [0,1] 范围的 CF 图像 tensor
    计算血管像素占比（简单二值化后取比例）。

    img_tensor_01: [B, 3, H, W] 或 [1, 3, H, W]，数值范围 [0,1]
    """
    vessel_map = extract_vessel_map(
        img_tensor_01.float(),
        image_type="cf",
        mode="cf2fa",  # 与 v22 训练时一致
    )

    # vessel_map 形状可能是 [B,1,H,W] 或 [B,H,W]，统一成单通道
    if vessel_map.dim() == 4:
        vm = vessel_map[0, 0]
    elif vessel_map.dim() == 3:
        vm = vessel_map[0]
    else:
        vm = vessel_map

    vm = vm.float()
    # 简单阈值分割，>0.5 视为血管
    ratio = (vm > 0.5).float().mean().item()
    return ratio


@torch.no_grad()
def compute_vessel_stats_on_cf_dir(cf_dir: str):
    """
    在真实 CF 目录上预统计血管像素占比的 min/max，用于后续生成样本的筛选。
    """
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    cf_paths = []
    for p in patterns:
        cf_paths.extend(glob.glob(os.path.join(cf_dir, p), recursive=True))
    cf_paths = sorted(list(set(cf_paths)))
    # 只保留以 "_01" 结尾的原始 CF 图，避免把 FA / mask 等其他模态也当成 CF
    cf_paths = [
        path for path in cf_paths
        if os.path.splitext(os.path.basename(path))[0].endswith("_01")
    ]

    if len(cf_paths) == 0:
        raise FileNotFoundError(f"在 cf_dir={cf_dir} 下未找到任何 PNG/JPG 图像")

    to_tensor = transforms.ToTensor()
    ratios = []

    for cf_path in cf_paths:
        img = Image.open(cf_path).convert("RGB")
        img = img.resize((SIZE, SIZE), Image.BICUBIC)
        img_tensor = to_tensor(img).unsqueeze(0).to(DEVICE)  # [0,1]
        ratio = compute_vessel_ratio_from_tensor(img_tensor)
        ratios.append(ratio)

    ratios_np = np.array(ratios, dtype=np.float32)
    min_ratio = float(ratios_np.min())
    max_ratio = float(ratios_np.max())

    return min_ratio, max_ratio, ratios_np


@torch.no_grad()
def generate_cf_images(unet, vae, tokenizer, text_encoder,
                       amount: int, out_dir: str, steps: int = 50):
    """
    从纯噪声生成若干 CF 图像。

    - unet: 已经加载好 LoRA 的 UNet（或其 base_model）
    - vae, tokenizer, text_encoder: 与 train_cf_gen.py 一致
    - amount: 需要生成的样本数量
    - out_dir: 输出根目录（对应某个 savedir），下级为 1,2,...,amount
    - steps: 采样步数（默认 50，与 train_cf_gen.py 中 visualize_random_cf 对齐）
    """
    os.makedirs(out_dir, exist_ok=True)

    # eval 模式以获得稳定推理
    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    # 文本 prompt：CF
    prompt_cf = get_cf_prompt_embeds(1, tokenizer, text_encoder)

    # 调度器与 train_cf_gen.py 保持一致
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    # 确定 latent 维度
    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )
    latent_shape = (1, in_channels, SIZE // 8, SIZE // 8)

    indices = range(1, amount + 1)
    if tqdm is not None:
        indices = tqdm(indices, desc="生成 CF 图像", ncols=80)

    for idx in indices:
        # 1. 采样初始噪声
        latents = torch.randn(latent_shape, device=DEVICE)

        # 2. 逐步去噪
        for t in scheduler.timesteps:
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_cf,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(
                    latents,
                    t,
                    prompt_cf,
                ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 3. VAE 解码
        latents_final = latents / vae.config.scaling_factor
        imgs_cf = vae.decode(latents_final).sample
        img_cf = tensor_to_pil(imgs_cf[0])

        # 4. 保存到 [out_dir]/[idx]/cf.png
        img_dir = os.path.join(out_dir, str(idx))
        os.makedirs(img_dir, exist_ok=True)
        img_cf.save(os.path.join(img_dir, "cf.png"))


@torch.no_grad()
def generate_cf_from_real_cf(
    unet,
    vae,
    tokenizer,
    text_encoder,
    cf_dir: str,
    out_dir: str,
    steps: int = 50,
    num_per_cf: int = 1,
    strength: float = 0.6,
    vessel_min: float | None = None,
    vessel_max: float | None = None,
    max_retries: int = 5,
):
    """
    从真实 CF 图出发，先编码到 latent，再在中间时间步加噪声后去噪，生成“结构相似但不完全相同”的增强样本。

    - cf_dir: 真实 CF 图所在目录（递归搜索 *.png, *.jpg, *.jpeg）
    - num_per_cf: 每张真实 CF 生成多少张增强图
    - strength: 噪声强度 (0~1]，越大越偏离原图，类似 SD img2img 的 strength
    """
    os.makedirs(out_dir, exist_ok=True)

    # 收集所有真实 CF 路径
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    cf_paths = []
    for p in patterns:
        cf_paths.extend(glob.glob(os.path.join(cf_dir, p), recursive=True))
    cf_paths = sorted(list(set(cf_paths)))
    # 只保留以 "_01" 结尾的原始 CF 图，避免把 FA / mask 等其他模态也当成 CF
    cf_paths = [
        path for path in cf_paths
        if os.path.splitext(os.path.basename(path))[0].endswith("_01")
    ]

    if len(cf_paths) == 0:
        raise FileNotFoundError(f"在 cf_dir={cf_dir} 下未找到任何 PNG/JPG 图像")

    # eval 模式
    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    # 文本 prompt：CF
    prompt_cf = get_cf_prompt_embeds(1, tokenizer, text_encoder)

    # 调度器，与 train_cf_gen.py 保持一致
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps  # 形如 [t_0, t_1, ..., t_{S-1}]，通常是降序（大→小）

    # 根据 strength 选择起始步，模仿 StableDiffusionImg2Img 的语义：
    # - strength 越大，加入的噪声越多，越接近纯生成
    # - strength 越小，越接近原图重建
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))
    num_t = len(timesteps)
    # 我们希望：strength=1.0 -> 使用 timesteps[0] (最大 t)；strength=0.0 -> timesteps[-1] (最小 t)
    start_index = int((1.0 - strength) * (num_t - 1))
    start_index = min(max(start_index, 0), num_t - 1)
    start_t = timesteps[start_index]

    to_tensor = transforms.ToTensor()
    idx = 1

    if tqdm is not None:
        cf_iter = tqdm(cf_paths, desc="从真实 CF 生成增强图像", ncols=80)
    else:
        cf_iter = cf_paths

    for cf_path in cf_iter:
        # 加载并预处理真实 CF（对齐训练时的 VAE 输入：RGB，SIZE×SIZE，[-1,1]）
        img = Image.open(cf_path).convert("RGB")
        img = img.resize((SIZE, SIZE), Image.BICUBIC)
        img_tensor = to_tensor(img).unsqueeze(0).to(DEVICE)  # [0,1]
        img_tensor = img_tensor * 2.0 - 1.0  # [-1,1]

        # VAE 编码
        latents_clean = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
        orig_stem = os.path.splitext(os.path.basename(cf_path))[0]

        for _ in range(max(1, num_per_cf)):
            retries = 0
            while retries < max_retries:

                # 在 start_t 上加噪
                noise = torch.randn_like(latents_clean)
                latents = scheduler.add_noise(latents_clean, noise, start_t)

                # 从 start_index 开始逆扩散到 0
                for t in timesteps[start_index:]:
                    if hasattr(unet, "base_model"):
                        noise_pred = unet.base_model(
                            sample=latents,
                            timestep=t,
                            encoder_hidden_states=prompt_cf,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = unet(
                            latents,
                            t,
                            prompt_cf,
                        ).sample

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # VAE 解码
                latents_final = latents / vae.config.scaling_factor
                imgs_cf = vae.decode(latents_final).sample
                img_cf = tensor_to_pil(imgs_cf[0])

                # 若设置了血管占比范围，进行筛选
                if vessel_min is not None and vessel_max is not None:
                    img_cf_tensor = to_tensor(img_cf).unsqueeze(0).to(DEVICE)  # [0,1]
                    ratio_cf = compute_vessel_ratio_from_tensor(img_cf_tensor)
                    if not (vessel_min <= ratio_cf <= vessel_max):
                        retries += 1
                        continue

                # 通过筛选或未启用筛选，则保存
                img_dir = os.path.join(out_dir, str(idx))
                os.makedirs(img_dir, exist_ok=True)
                # 增强后的 CF
                img_cf.save(os.path.join(img_dir, "cf.png"))
                # 对应的原始 CF（resize 到 512，一起作为配对）
                orig_name = f"{orig_stem}_cf_origin.png"
                img.save(os.path.join(img_dir, orig_name))

                idx += 1
                break

            if retries >= max_retries:
                print(f"  [警告] {cf_path} 的一个增强样本在 {max_retries} 次内未生成合格血管占比，已跳过。")


# ============ 主函数 ============


def main():
    parser = argparse.ArgumentParser(description="V22 CF 生成模型推理脚本")
    parser.add_argument(
        "-n", "--name", default="cf_gen_v22",
        help="训练时使用的实验名称（与 train_cf_gen.py 的 --name 对应）",
    )
    parser.add_argument(
        "--amount", type=int, default=1000,
        help="需要生成的 CF 样本数量（默认 1000）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="noise",
        choices=["noise", "from_cf"],
        help="生成模式：noise=从纯噪声生成；from_cf=从真实 CF 加噪声再去噪做增强",
    )
    parser.add_argument(
        "--savedir", type=str, default="1",
        help="保存路径编号，将作为二级目录名，例如 '1'、'001' 等",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="",
        help="可选，自定义 checkpoint 目录；为空时使用默认 best_checkpoint。"
    )
    parser.add_argument(
        "--cf_dir",
        type=str,
        default="",
        help="mode=from_cf 时：真实 CF 图所在目录（递归读取 *.png,*.jpg,*.jpeg）",
    )
    parser.add_argument(
        "--num_per_cf",
        type=int,
        default=1,
        help="mode=from_cf 时：每张真实 CF 生成多少张增强图（默认 1）",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="mode=from_cf 时的噪声强度 (0~1]，越大越偏离原图，类似 img2img strength，默认 0.6",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="扩散采样步数，默认 50，与训练时可视化一致",
    )
    parser.add_argument(
        "--vessel_retry", type=int, default=5,
        help="mode=from_cf 时，单个样本为满足血管占比范围时的最大重试次数（默认 5）",
    )
    args = parser.parse_args()

    # 1. 准备输出目录
    out_dir = os.path.join(PRED_OUT_ROOT, args.name, str(args.savedir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== CF 生成推理 (V22) =====")
    print(f"  - 实验名称: {args.name}")
    print(f"  - 生成数量: {args.amount}")
    print(f"  - 保存目录: {out_dir}")

    # 2. 加载基础模型（与 train_cf_gen.py 对齐）
    print("\n[1/3] 加载基础 SD1.5 模型...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="text_encoder"
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_DIR, subfolder="vae"
    ).to(DEVICE)
    unet_base = UNet2DConditionModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="unet"
    ).to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_base.requires_grad_(False)

    # 3. 加载训练好的 UNet LoRA
    #    - 默认：使用 best_checkpoint
    #    - 如果提供了 --ckpt_dir：优先使用该目录
    print("\n[2/3] 加载训练好的 UNet LoRA ...")
    if args.ckpt_dir:
        # 支持两种形式：
        # 1) 传 step_XXXXXX 目录：里面有 unet_lora 子目录
        # 2) 直接传 unet_lora 目录
        base_dir = args.ckpt_dir
        candidate = os.path.join(base_dir, "unet_lora")
        if os.path.isdir(candidate):
            lora_dir = candidate
        else:
            lora_dir = base_dir
        print(f"  - 使用自定义 ckpt_dir: {args.ckpt_dir}")
    else:
        lora_dir = os.path.join(
            TRAIN_OUT_ROOT, args.name, "best_checkpoint", "unet_lora"
        )
        print(f"  - 未指定 ckpt_dir，使用默认 best_checkpoint: {lora_dir}")

    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"未找到 LoRA 权重目录: {lora_dir}\n"
            f"若使用默认模式，请确认 `train_cf_gen.py` 已经训练完成并保存了 best_checkpoint；\n"
            f"若使用 --ckpt_dir，请确保该目录本身是 unet_lora，或其下包含 unet_lora 子目录。"
        )

    # 使用 PEFT 的 PeftModel.from_pretrained 加载 LoRA 适配器
    unet_lora = PeftModel.from_pretrained(unet_base, lora_dir)
    print(f"  ✓ LoRA 权重已从 {lora_dir} 加载")

    # 4. 选择生成模式
    if args.mode == "noise":
        print("\n[3/3] 开始从纯噪声生成 CF 图像...")
        generate_cf_images(
            unet=unet_lora,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            amount=args.amount,
            out_dir=out_dir,
            steps=args.steps,
        )
        total_gen = args.amount
    else:
        if not args.cf_dir:
            raise ValueError("mode=from_cf 时必须指定 --cf_dir 作为真实 CF 图像目录")
        print("\n[2.5/3] 预统计真实 CF 的血管像素占比范围...")
        vessel_min, vessel_max, vessel_ratios = compute_vessel_stats_on_cf_dir(args.cf_dir)
        print(f"  - 样本数: {len(vessel_ratios)}")
        print(f"  - 血管像素占比 min={vessel_min:.4f}, max={vessel_max:.4f}")
        with open(os.path.join(out_dir, "vessel_ratio_stats.txt"), "w", encoding="utf-8") as f:
            f.write(f"num_samples: {len(vessel_ratios)}\n")
            f.write(f"min_ratio: {vessel_min:.6f}\n")
            f.write(f"max_ratio: {vessel_max:.6f}\n")

        print("\n[3/3] 开始从真实 CF 加噪声生成增强 CF 图像，并按血管占比范围过滤...")
        print(f"  - cf_dir: {args.cf_dir}")
        print(f"  - num_per_cf: {args.num_per_cf}")
        print(f"  - strength: {args.strength}")
        generate_cf_from_real_cf(
            unet=unet_lora,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            cf_dir=args.cf_dir,
            out_dir=out_dir,
            steps=args.steps,
            num_per_cf=args.num_per_cf,
            strength=args.strength,
            vessel_min=vessel_min,
            vessel_max=vessel_max,
            max_retries=args.vessel_retry,
        )
        # 无法准确预知多少张，提示用户到目录查看
        total_gen = None

    if total_gen is not None:
        print(f"\n✓ 推理完成，共生成 {total_gen} 张 CF 图像。")
    else:
        print(f"\n✓ 推理完成。请查看目录 {out_dir} 下生成的 CF 图像数量。")
    print(f"  输出路径: {out_dir}")


if __name__ == "__main__":
    main()

