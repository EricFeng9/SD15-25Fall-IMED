import os, csv, time, argparse
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel)

# 基本配置（v3：SD2.1-768 + ControlNet SD2.1 HED）
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd21-diffusers"
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v3.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v3.csv"  # 如需验证可复用
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd21"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练分辨率（与 sd2.1-768 对齐）
SIZE = 768

# 可选：LPIPS（有则计算，无则跳过）
try:
    import lpips  # pip install lpips
    _has_lpips = True
except Exception:
    _has_lpips = False
    lpips = None


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    """
    将 PIL 图像转换为张量 RGB，缩放到固定分辨率 SIZE×SIZE，数值范围为 [0,1]。
    """
    t = transforms.ToTensor()(img.convert("RGB"))
    t = transforms.functional.resize(t, [SIZE, SIZE])
    return t


def _pick_paths(row: dict, mode: str):
    """
    从一行 CSV 中取出 cond(条件图) 与 tgt(目标图) 的路径。
    仅依赖两列：cf_path、octa_path。

    - 当 mode = "cf2octa"：cond=CF，tgt=OCTA
    - 当 mode = "octa2cf"：cond=OCTA，tgt=CF
    """
    cf_path = row.get("cf_path")
    octa_path = row.get("octa_path")
    if not cf_path or not octa_path:
        raise ValueError("CSV 需要同时包含 cf_path 与 octa_path 两列")
    if mode == "cf2octa":
        return cf_path, octa_path
    else:
        return octa_path, cf_path


class PairCSV(Dataset):
    """
    简单的成对数据集：读取 CSV，每个样本返回 (cond, tgt)。
    - cond: ControlNet 条件输入（[0,1] RGB）
    - tgt : 目标域图像，经线性缩放到 [-1,1]（因为 VAE 期望该范围）
    """
    def __init__(self, csv_path: str, mode: str):
        self.rows = []
        self.mode = mode
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd:
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        src_path, dst_path = _pick_paths(r, self.mode)
        cond = pil_to_tensor_rgb(Image.open(src_path))
        tgt  = pil_to_tensor_rgb(Image.open(dst_path))
        # VAE 输入需 [-1,1]
        tgt = tgt * 2 - 1
        return cond, tgt


# 文本与 VAE 编码工具（依赖全局 pipe/vae_sf 在 main() 中初始化）
def get_prompt_embeds(bs: int) -> torch.Tensor:
    """
    生成空 prompt 的文本嵌入，用于条件生成。返回形状 [bs, seq, dim]。
    """
    prompts = [""] * bs
    tok = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        te_out = pipe.text_encoder(tok.input_ids.to(device))
        prompt_embeds = te_out[0]
    return prompt_embeds


def encode_vae(img: torch.Tensor) -> torch.Tensor:
    """将图像张量（[-1,1]）编码到 VAE 潜空间，并乘以 scaling_factor。"""
    latents = pipe.vae.encode(img).latent_dist.sample() * vae_sf
    return latents


# -------------------------
# 评估指标：SSIM / MS-SSIM（纯 PyTorch 实现）
# -------------------------

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    w = w / w.sum()
    w = w.view(1, 1, window_size, window_size)
    w = w.repeat(channels, 1, 1, 1)
    return w


def _ssim_map(x, y, window, data_range=1.0, K1=0.01, K2=0.03):
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    padding = window.shape[-1] // 2
    mu1 = torch.nn.functional.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu2 = torch.nn.functional.conv2d(y, window, padding=padding, groups=y.shape[1])

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return ssim_map, cs_map


def ssim(x, y, window_size=11, sigma=1.5):
    # 期望 x,y ∈ [0,1]，形状 [N,C,H,W]
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    window = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    ssim_map, _ = _ssim_map(x, y, window)
    return ssim_map.mean().item()


def ms_ssim(x, y, window_size=11, sigma=1.5, weights=None, levels=5):
    # 经典 MS-SSIM 权重
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    window = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    mcs = []
    for _ in range(levels - 1):
        ssim_map, cs_map = _ssim_map(x, y, window)
        mcs.append(cs_map.mean(dim=(1, 2, 3)))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        y = torch.nn.functional.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
    ssim_map, _ = _ssim_map(x, y, window)
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    mcs = torch.stack(mcs, dim=0)
    weights_t = torch.tensor(weights, device=ssim_val.device, dtype=ssim_val.dtype)
    ms = torch.prod(mcs**weights_t[:-1].view(-1, 1), dim=0) * (ssim_val ** weights_t[-1])
    return ms.mean().item()


def _compute_metrics(pil_pred: Image.Image, pil_gt: Image.Image, metric_state: dict):
    """
    计算 LPIPS / SSIM / MS-SSIM。
    - pil_pred, pil_gt: PIL RGB，已同尺寸。
    - metric_state: 内部缓存（如 lpips 模型）
    返回: dict
    """
    # 转为张量 [0,1]
    x = pil_to_tensor_rgb(pil_pred).unsqueeze(0).to(device)
    y = pil_to_tensor_rgb(pil_gt).unsqueeze(0).to(device)

    # SSIM / MS-SSIM（在 [0,1] 上计算）
    ssim_val = ssim(x, y)
    ms_val = ms_ssim(x, y)

    # LPIPS（可选；需要 [-1,1]）
    lpips_val = None
    if metric_state.get("lpips_fn") is not None:
        with torch.no_grad():
            xx = x * 2 - 1
            yy = y * 2 - 1
            lp = metric_state["lpips_fn"](xx, yy).mean().item()
            lpips_val = float(lp)
    return {"lpips": lpips_val, "ssim": float(ssim_val), "ms_ssim": float(ms_val)}


def _run_eval_and_log(eval_rows, step_dir, mode, metric_state):
    """
    在给定 eval_rows 上运行推理与指标计算，将均值写入 step_dir/eval.txt。
    """
    if not eval_rows:
        return
    os.makedirs(step_dir, exist_ok=True)
    lpips_list, ssim_list, msssim_list = [], [], []
    for r in eval_rows:
        src_path, dst_path = _pick_paths(r, mode)
        cond_pil = Image.open(src_path).convert("RGB").resize((SIZE, SIZE))
        with torch.no_grad():
            pred_pil = pipe(
                prompt="",
                image=cond_pil,
                num_inference_steps=30,
                guidance_scale=3.5,
                controlnet_conditioning_scale=0.8
            ).images[0]
        gt_pil = Image.open(dst_path).convert("RGB").resize((SIZE, SIZE))
        m = _compute_metrics(pred_pil, gt_pil, metric_state)
        if m["lpips"] is not None:
            lpips_list.append(m["lpips"])
        ssim_list.append(m["ssim"])
        msssim_list.append(m["ms_ssim"])
    avg_lpips = (sum(lpips_list)/len(lpips_list)) if lpips_list else None
    avg_ssim = sum(ssim_list)/len(ssim_list)
    avg_msssim = sum(msssim_list)/len(msssim_list)
    with open(os.path.join(step_dir, "eval.txt"), "a") as f:
        f.write(f"lpips_mean={avg_lpips}\n")
        f.write(f"ssim_mean={avg_ssim}\n")
        f.write(f"ms_ssim_mean={avg_msssim}\n")
    print("评估均值 ->", "LPIPS:" , avg_lpips, "SSIM:", avg_ssim, "MS-SSIM:", avg_msssim)


def main():
    # 解析参数：仅保留必要参数，保证脚本简洁
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                        help="任务方向：cf2octa 表示 CF→OCTA，octa2cf 表示 OCTA→CF")
    parser.add_argument("--name", default='v3_cf_input',
                        help="实验名称：用于组织输出目录")
    parser.add_argument("--train_csv", default=train_csv,
                        help="训练用 CSV 路径（仅两列 cf_path, octa_path）")
    parser.add_argument("--eval_csv", default=val_csv,
                        help="训练后评估用 CSV（可用测试对）")
    parser.add_argument("--eval_n", type=int, default=20,
                        help="评估时取前 N 个样本计算指标")
    args, _ = parser.parse_known_args()

    # 输出目录：按 mode/name 组织
    v3_tag = args.name if args.name else 'v3_cf_input'
    out_dir = os.path.join(out_root, args.mode, v3_tag)
    os.makedirs(out_dir, exist_ok=True)

    # 数据加载（仅训练集；如需验证可自行添加 DataLoader）
    train_ds = PairCSV(args.train_csv, mode=args.mode)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    # 预读取评估样本（复用）
    eval_rows = []
    try:
        with open(args.eval_csv) as f:
            for i, r in enumerate(csv.DictReader(f)):
                if i >= args.eval_n:
                    break
                eval_rows.append(r)
    except Exception as e:
        print("评估预加载失败，将跳过周期评估：", e)
        eval_rows = []

# 模型组件（SD2.1 + ControlNet HED）
    os.environ["HF_HUB_OFFLINE"] = "1"  # 仅离线加载

    # 尝试从目录加载 ControlNet；若失败则尝试单文件 .safetensors（兼容不同权重格式）
ctrl_path = ctrl_dir
ctrl_model = None
try:
    ctrl_model = ControlNetModel.from_pretrained(ctrl_path, local_files_only=True)
except Exception:
    safes = [p for p in os.listdir(ctrl_path) if p.endswith('.safetensors')]
    if safes:
        try:
                ctrl_model = ControlNetModel.from_single_file(os.path.join(ctrl_path, safes[0]), local_files_only=True)
        except Exception as e:
            raise e
    else:
        raise

    global pipe, vae_sf
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_dir,
    controlnet=ctrl_model,
    local_files_only=True
).to(device)
pipe.enable_attention_slicing("max")
pipe.vae.enable_tiling()

    # 冻结 SD 主干，只训练 ControlNet
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.controlnet.requires_grad_(True)

noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
opt = torch.optim.AdamW(pipe.controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
    global vae_sf
    vae_sf = pipe.vae.config.scaling_factor
mse = nn.MSELoss()

    # 初始化 LPIPS（可选，复用）
    metric_state = {"lpips_fn": None}
    if _has_lpips:
        try:
            metric_state["lpips_fn"] = lpips.LPIPS(net='vgg').to(device).eval()
        except Exception as e:
            print("LPIPS 初始化失败，跳过：", e)

    max_steps = 5000
global_step = 0
pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
pipe.controlnet.train()

    # 计时：用于统计每 100 step 耗时
if device.type == "cuda":
    torch.cuda.synchronize()
t_block = time.time()

    print(f"\n[v3] 模型加载完成（SD2.1-768 + ControlNet HED），开始训练... 模式: {args.mode} | 输出: {out_dir} | 训练CSV(两列): {args.train_csv}")

while global_step < max_steps:
    for cond, tgt in train_loader:
            if global_step >= max_steps:
                break
        cond = cond.to(device)
        tgt  = tgt.to(device)
        b = tgt.shape[0]

        with torch.no_grad():
            latents = encode_vae(tgt)  # [b,4,H/8,W/8]
            noise   = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds = get_prompt_embeds(b)
            cond_img = cond

            # 控制分支前向（SD2.1 不需要 added_cond_kwargs）
        down_samples, mid_sample = pipe.controlnet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            controlnet_cond=cond_img, return_dict=False
        )

        # UNet 预测噪声
        noise_pred = pipe.unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample
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
            print(f"[v3] step {global_step}/{max_steps} loss {loss.item():.4f} | 100step {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_block = time.time()
        if global_step % 1000 == 0:
            # 保存快照
                step_dir = os.path.join(out_dir, f"step_{global_step}")
                pipe.controlnet.save_pretrained(step_dir)
                # 周期评估（若有 eval_rows）
                if eval_rows:
                    print(f"[v3] 进行周期评估 @ step {global_step}，样本数={len(eval_rows)}")
                    _run_eval_and_log(eval_rows, step_dir, args.mode, metric_state)

# 最终保存
pipe.controlnet.save_pretrained(out_dir)
print("[v3] saved to", out_dir) 

    # -------------------------
    # 训练后评估（小样本）
    # -------------------------
    if eval_rows:
        print(f"开始评估样本数: {len(eval_rows)}")
        _run_eval_and_log(eval_rows, out_dir, args.mode, metric_state)


if __name__ == "__main__":
    main()
