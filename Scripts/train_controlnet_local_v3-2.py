import os, csv, time, argparse
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel)

# 新增：Frangi 条件构建依赖
import numpy as np
try:
    from skimage.filters import frangi
    from skimage import exposure
    _has_skimage = True
except Exception:
    _has_skimage = False

# 基本配置（v3-2：SD2.1-768 + ControlNet SD2.1；条件统一为 Frangi）
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd21-diffusers"
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v3.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v3.csv"
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd21_v3_2"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 768


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    t = transforms.ToTensor()(img.convert("RGB"))
    t = transforms.functional.resize(t, [SIZE, SIZE])
    return t

# 保存条件图到文件（把 [-1,1] 张量还原到 [0,1] 并转 PIL）
def _save_cond_image(cond_tensor: torch.Tensor, save_path: str):
    c = cond_tensor.detach().float().cpu()
    if c.dim() == 4:
        c = c[0]
    c = c.clamp(-1, 1)
    c01 = (c + 1) / 2
    img = transforms.ToPILImage()(c01)
    img.save(save_path)

# 保存原始源图（[0,1] 张量转 PIL）
def _save_src_image(src_tensor: torch.Tensor, save_path: str):
    x = src_tensor.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0, 1)
    img = transforms.ToPILImage()(x)
    img.save(save_path)

# 保存 Frangi 灰度图（[0,1] 单通道张量转灰度 PIL）
def _save_gray_image(gray_tensor: torch.Tensor, save_path: str):
    g = gray_tensor.detach().float().cpu()
    if g.dim() == 4:
        g = g[0]
    if g.dim() == 3 and g.shape[0] > 1:
        g = g[0:1]
    if g.dim() == 2:
        g = g.unsqueeze(0)
    g = g.clamp(0, 1)
    img = transforms.ToPILImage()(g)
    img.save(save_path)


def _pick_paths(row: dict, mode: str):
    cf_path = row.get("cf_path"); octa_path = row.get("octa_path")
    if not cf_path or not octa_path:
        raise ValueError("CSV 需要同时包含 cf_path 与 octa_path 两列")
    if mode == "cf2octa":
        return cf_path, octa_path
    else:
        return octa_path, cf_path


# ---------- Frangi 条件生成 ----------
# 返回规范化到 [0,1] 的 Frangi 灰度二维数组（numpy）
def _frangi_gray_from_pil(src_pil: Image.Image) -> np.ndarray:
    if not _has_skimage:
        raise ImportError("未检测到 scikit-image，请先安装：pip install scikit-image")
    gray = np.array(src_pil.convert("L"), dtype=np.float32) / 255.0
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.01)
    v = frangi(gray_eq)
    v = v - v.min(); vmax = v.max(); v = v / (vmax + 1e-8)
    return v

# 将 [0,1] 灰度二维数组拼成 3 通道 RGB PIL
def _rgb_from_gray(v_gray: np.ndarray) -> Image.Image:
    v8 = (v_gray * 255.0).astype(np.uint8)
    v3 = np.stack([v8, v8, v8], axis=-1)
    return Image.fromarray(v3, mode="RGB")


def _frangi_from_pil(src_pil: Image.Image) -> Image.Image:
    v = _frangi_gray_from_pil(src_pil)
    return _rgb_from_gray(v)


class PairCSVFrangi(Dataset):
    """
    读取 CSV，每个样本返回 (cond_frangi, tgt, src_rgb_tensor, frangi_gray_tensor)：
    - cond_frangi      : Frangi 增强后的血管结构 RGB 条件图，[-1,1]
    - tgt              : 目标域图像，[-1,1]
    - src_rgb_tensor   : 源图 RGB，[0,1]
    - frangi_gray_tensor: Frangi 灰度单通道，[0,1]
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
        src_pil = Image.open(src_path).convert("RGB").resize((SIZE, SIZE))
        # Frangi 条件
        fr_gray = _frangi_gray_from_pil(src_pil)
        fr_pil = _rgb_from_gray(fr_gray)
        cond = pil_to_tensor_rgb(fr_pil)
        cond = cond * 2 - 1
        # 目标
        tgt  = pil_to_tensor_rgb(Image.open(dst_path))
        tgt  = tgt * 2 - 1
        # 调试用：源图 & Frangi 灰度
        src_rgb_tensor = pil_to_tensor_rgb(src_pil)              # [0,1]
        frangi_gray_tensor = torch.from_numpy(fr_gray).float()   # [H,W]
        frangi_gray_tensor = frangi_gray_tensor.unsqueeze(0)     # [1,H,W]
        return cond, tgt, src_rgb_tensor, frangi_gray_tensor


# 文本与 VAE 编码工具（依赖全局 pipe/vae_sf 在 main() 中初始化）
def get_prompt_embeds(bs: int) -> torch.Tensor:
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
    latents = pipe.vae.encode(img).latent_dist.sample() * vae_sf
    return latents


# 简单评估（LPIPS 可选；SSIM/MS-SSIM 纯 PyTorch）
try:
    import lpips
    _has_lpips = True
except Exception:
    _has_lpips = False
    lpips = None

import torch.nn.functional as F

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma)); g = g / g.sum()
    w = g[:, None] * g[None, :]; w = w / w.sum()
    w = w.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return w


def _ssim_map(x, y, window, data_range=1.0, K1=0.01, K2=0.03):
    C1 = (K1 * data_range) ** 2; C2 = (K2 * data_range) ** 2
    pad = window.shape[-1] // 2
    mu1 = F.conv2d(x, window, padding=pad, groups=x.shape[1])
    mu2 = F.conv2d(y, window, padding=pad, groups=y.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sig1 = F.conv2d(x*x, window, padding=pad, groups=x.shape[1]) - mu1_sq
    sig2 = F.conv2d(y*y, window, padding=pad, groups=y.shape[1]) - mu2_sq
    sig12= F.conv2d(x*y, window, padding=pad, groups=x.shape[1]) - mu1_mu2
    ssim_n = (2*mu1_mu2 + C1) * (2*sig12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    cs_map = (2*sig12 + C2) / (sig1 + sig2 + C2)
    return ssim_map, cs_map


def ssim(x, y, window_size=11, sigma=1.5):
    x = x.clamp(0,1); y = y.clamp(0,1)
    w = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    s,_ = _ssim_map(x, y, w)
    return s.mean().item()


def ms_ssim(x, y, window_size=11, sigma=1.5, weights=None, levels=5):
    if weights is None: weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    x = x.clamp(0,1); y = y.clamp(0,1)
    w = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    mcs = []
    for _ in range(levels-1):
        s, cs = _ssim_map(x, y, w); mcs.append(cs.mean(dim=(1,2,3)))
        x = F.avg_pool2d(x, 2); y = F.avg_pool2d(y, 2)
    s,_ = _ssim_map(x, y, w); sv = s.mean(dim=(1,2,3))
    mcs = torch.stack(mcs, dim=0)
    wt = torch.tensor(weights, device=sv.device, dtype=sv.dtype)
    ms = torch.prod(mcs**wt[:-1].view(-1,1), dim=0) * (sv ** wt[-1])
    return ms.mean().item()


def _compute_metrics(pil_pred: Image.Image, pil_gt: Image.Image, metric_state: dict):
    x = pil_to_tensor_rgb(pil_pred).unsqueeze(0).to(device)
    y = pil_to_tensor_rgb(pil_gt).unsqueeze(0).to(device)
    ssim_val = ssim(x, y); ms_val = ms_ssim(x, y)
    lpips_val = None
    if metric_state.get("lpips_fn") is not None:
        with torch.no_grad():
            xx = x*2-1; yy = y*2-1
            lpips_val = float(metric_state["lpips_fn"](xx, yy).mean().item())
    return {"lpips": lpips_val, "ssim": float(ssim_val), "ms_ssim": float(ms_val)}


def _run_eval_and_log(eval_rows, step_dir, mode, metric_state):
    if not eval_rows: return
    os.makedirs(step_dir, exist_ok=True)
    lp, s1, s2 = [], [], []
    for r in eval_rows:
        src_path, dst_path = _pick_paths(r, mode)
        src = Image.open(src_path).convert("RGB").resize((SIZE,SIZE))
        cond = _frangi_from_pil(src)
        with torch.no_grad():
            pred = pipe(prompt="", image=cond, num_inference_steps=30,
                        guidance_scale=3.5, controlnet_conditioning_scale=0.8).images[0]
        gt = Image.open(dst_path).convert("RGB").resize((SIZE,SIZE))
        m = _compute_metrics(pred, gt, metric_state)
        if m["lpips"] is not None: lp.append(m["lpips"])
        s1.append(m["ssim"]); s2.append(m["ms_ssim"])
    avg_lp = (sum(lp)/len(lp)) if lp else None
    avg_s = sum(s1)/len(s1); avg_ms = sum(s2)/len(s2)
    with open(os.path.join(step_dir, "eval.txt"), "a") as f:
        f.write(f"lpips_mean={avg_lp}\n"); f.write(f"ssim_mean={avg_s}\n"); f.write(f"ms_ssim_mean={avg_ms}\n")
    print("评估均值 ->", "LPIPS:", avg_lp, "SSIM:", avg_s, "MS-SSIM:", avg_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
    parser.add_argument("--name", default='v3_2_frangi')
    parser.add_argument("--train_csv", default=train_csv)
    parser.add_argument("--eval_csv", default=val_csv)
    parser.add_argument("--eval_n", type=int, default=20)
    args, _ = parser.parse_known_args()

    if not _has_skimage:
        raise ImportError("本脚本需要 scikit-image，请先安装：pip install scikit-image")

    vtag = args.name if args.name else 'v3_2_frangi'
    out_dir = os.path.join(out_root, args.mode, vtag); os.makedirs(out_dir, exist_ok=True)

    # 数据加载（仅训练集）
    train_ds = PairCSVFrangi(args.train_csv, mode=args.mode)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    # 仅对 Diffusers 组件启用离线
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 加载 ControlNet（作为初始化权重）
    ctrl_model = ControlNetModel.from_pretrained(ctrl_dir, local_files_only=True)

    global pipe, vae_sf
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_dir, controlnet=ctrl_model, local_files_only=True
    ).to(device)
    pipe.enable_attention_slicing("max"); pipe.vae.enable_tiling()

    # 冻结主干，只训 ControlNet
    pipe.unet.requires_grad_(False); pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    pipe.controlnet.requires_grad_(True)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    opt = torch.optim.AdamW(pipe.controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
    vae_sf = pipe.vae.config.scaling_factor
    mse = nn.MSELoss()

    # LPIPS（可选）
    metric_state = {"lpips_fn": None}
    if _has_lpips:
        try:
            metric_state["lpips_fn"] = lpips.LPIPS(net='vgg').to(device).eval()
        except Exception as e:
            print("LPIPS 初始化失败，跳过：", e)

    max_steps = 10000; global_step = 0
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval(); pipe.controlnet.train()

    if device.type == "cuda": torch.cuda.synchronize()
    t_block = time.time()
    print(f"\n[v3-2] 统一 Frangi 条件，开始训练... 模式: {args.mode} | 输出: {out_dir} | 训练CSV: {args.train_csv}")

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps: break
            # 解包新增的源图与 Frangi 灰度
            cond, tgt, src_rgb_tensor, frangi_gray_tensor = batch
            cond = cond.to(device); tgt = tgt.to(device); b = tgt.shape[0]
            with torch.no_grad():
                latents = encode_vae(tgt); noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prompt_embeds = get_prompt_embeds(b); cond_img = cond
            down_samples, mid_sample = pipe.controlnet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_img, return_dict=False
            )
            noise_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample
            ).sample
            loss = mse(noise_pred, noise)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            global_step += 1
            if global_step % 100 == 0:
                if device.type == "cuda": torch.cuda.synchronize()
                elapsed = time.time() - t_block
                print(f"[v3-2] step {global_step}/{max_steps} loss {loss.item():.4f} | 100step {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
                if device.type == "cuda": torch.cuda.synchronize(); t_block = time.time()
            if global_step % 1000 == 0:
                step_dir = os.path.join(out_dir, f"step_{global_step}"); pipe.controlnet.save_pretrained(step_dir)
                # 保存一张本轮条件图与源图、Frangi 灰度到权重目录
                try:
                    _save_cond_image(cond, os.path.join(step_dir, "cond_frangi.png"))
                except Exception as e:
                    pass
                try:
                    _save_src_image(src_rgb_tensor, os.path.join(step_dir, "src.png"))
                except Exception as e:
                    pass
                try:
                    _save_gray_image(frangi_gray_tensor, os.path.join(step_dir, "cond_frangi_gray.png"))
                except Exception as e:
                    pass
                # 周期评估（统一 Frangi）
                print(f"[v3-2] 进行周期评估 @ step {global_step}")
                # 预载 eval 样本
                eval_rows = []
                try:
                    with open(args.eval_csv) as f:
                        for i, r in enumerate(csv.DictReader(f)):
                            if i >= args.eval_n: break
                            eval_rows.append(r)
                except Exception as e:
                    eval_rows = []
                if eval_rows:
                    _run_eval_and_log(eval_rows, step_dir, args.mode, metric_state)

    pipe.controlnet.save_pretrained(out_dir)
    print("[v3-2] saved to", out_dir)

    # 训练后评估（统一 Frangi）
    try:
        eval_rows = []
        with open(args.eval_csv) as f:
            for i, r in enumerate(csv.DictReader(f)):
                if i >= args.eval_n: break
                eval_rows.append(r)
        if eval_rows:
            _run_eval_and_log(eval_rows, out_dir, args.mode, metric_state)
    except Exception as e:
        pass


if __name__ == "__main__":
    main() 