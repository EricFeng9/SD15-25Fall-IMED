import os, csv, argparse
import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 仅离线加载权重
os.environ["HF_HUB_OFFLINE"] = "1"

# 基本配置
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd21-diffusers"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v3.csv"  # 仅两列 cf_path, octa_path
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd21"
SIZE = 768

# 可选 LPIPS
try:
    import lpips
    _has_lpips = True
except Exception:
    _has_lpips = False
    lpips = None


def _load_controlnet(ctrl_dir: str):
    """
    加载 ControlNet：优先按目录加载；失败则尝试目录下单文件 .safetensors。
    """
    try:
        return ControlNetModel.from_pretrained(ctrl_dir, torch_dtype=torch.float16, local_files_only=True)
    except Exception:
        safes = [p for p in os.listdir(ctrl_dir) if p.endswith('.safetensors')]
        if not safes:
            raise
        try:
            return ControlNetModel.from_single_file(os.path.join(ctrl_dir, safes[0]), torch_dtype=torch.float16)
        except Exception as e:
            raise e


def _pick_src(row: dict, mode: str) -> str:
    """
    从 CSV 行选取输入条件图路径：
    - cf2octa: 使用 cf_path
    - octa2cf: 使用 octa_path
    CSV 仅包含这两列。
    """
    if mode == "cf2octa":
        return row.get("cf_path")
    else:
        return row.get("octa_path")


# -------- SSIM / MS-SSIM 与 LPIPS 计算 --------
import torchvision.transforms.functional as TF

def _pil_to_tensor_01(img: Image.Image):
    t = TF.to_tensor(img.convert("RGB"))
    t = TF.resize(t, [SIZE, SIZE])
    return t.unsqueeze(0)


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
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    window = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    ssim_map, _ = _ssim_map(x, y, window)
    return ssim_map.mean().item()


def ms_ssim(x, y, window_size=11, sigma=1.5, weights=None, levels=5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    window = _gaussian_window(window_size, sigma, x.shape[1], x.device, x.dtype)
    mcs = []
    for _ in range(levels - 1):
        ssim_map, cs_map = _ssim_map(x, y, window)
        mcs.append(cs_map.mean(dim=(1, 2, 3)))
        x = torch.nn.functional.avg_pool2d(x, 2)
        y = torch.nn.functional.avg_pool2d(y, 2)
    ssim_map, _ = _ssim_map(x, y, window)
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    mcs = torch.stack(mcs, dim=0)
    weights_t = torch.tensor(weights, device=ssim_val.device, dtype=ssim_val.dtype)
    ms = torch.prod(mcs**weights_t[:-1].view(-1, 1), dim=0) * (ssim_val ** weights_t[-1])
    return ms.mean().item()


def _compute_metrics(pred_pil: Image.Image, gt_pil: Image.Image, lpips_fn, device):
    x = _pil_to_tensor_01(pred_pil).to(device)
    y = _pil_to_tensor_01(gt_pil).to(device)
    ssim_val = ssim(x, y)
    mssim_val = ms_ssim(x, y)
    lp = None
    if lpips_fn is not None:
        with torch.no_grad():
            xx = x * 2 - 1
            yy = y * 2 - 1
            lp = float(lpips_fn(xx, yy).mean().item())
    return lp, ssim_val, mssim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                        help="任务方向：cf2octa 表示 CF→OCTA，octa2cf 表示 OCTA→CF")
    parser.add_argument("--name", default="default",
                        help="权重名称：用于组织 ctrl_dir 与输出目录")
    parser.add_argument("--ctrl_dir", default=None,
                        help="可选：直接指定 ControlNet 目录（优先级最高）")
    parser.add_argument("--csv", default=csv_path,
                        help="推理 CSV 路径（仅两列 cf_path, octa_path）")
    parser.add_argument("--step", type=int, default=None,
                        help="可选：从 {ctrl_root}/{mode}/{name}/step_{N} 选择子目录")
    parser.add_argument("--savedir", default=None,
                        help="可选：结果保存到 {out_root}/{mode}/{name}/{savedir}")
    # 推理参数
    parser.add_argument("--prompt", default="", help="正向提示词")
    parser.add_argument("--negative_prompt", default="", help="负向提示词")
    parser.add_argument("--cn_scale", type=float, default=0.8, help="ControlNet 条件强度")
    parser.add_argument("--cfg", type=float, default=3.5, help="文本引导强度（CFG）")
    parser.add_argument("--steps", type=int, default=30, help="去噪步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    args = parser.parse_args()

    # 解析 ControlNet 权重目录
    if args.ctrl_dir:
        ctrl_dir = args.ctrl_dir
    else:
        base_ctrl = os.path.join(ctrl_root, args.mode, args.name if args.name else "default")
        ctrl_dir = os.path.join(base_ctrl, f"step_{args.step}") if args.step else base_ctrl
    if not os.path.isdir(ctrl_dir):
        raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

    # 输出目录
    base_out = os.path.join(out_root, args.mode, args.name if args.name else "default")
    if args.savedir:
        out_dir = os.path.join(base_out, args.savedir)
    else:
        out_dir = os.path.join(base_out, f"step_{args.step}") if args.step else base_out
    os.makedirs(out_dir, exist_ok=True)

    # 确定随机种子
    used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))

    # 写入推理参数日志
    log_path = os.path.join(out_dir, "log.txt")
    with open(log_path, "a") as _f:
        _f.write("inference_params\n")
        _f.write(f"mode={args.mode}\n")
        _f.write(f"name={args.name}\n")
        _f.write(f"ctrl_dir={ctrl_dir}\n")
        _f.write(f"csv={args.csv}\n")
        _f.write(f"step={args.step}\n")
        _f.write(f"savedir={args.savedir}\n")
        _f.write(f"prompt={args.prompt}\n")
        _f.write(f"negative_prompt={args.negative_prompt}\n")
        _f.write(f"cn_scale={args.cn_scale}\n")
        _f.write(f"cfg={args.cfg}\n")
        _f.write(f"steps={args.steps}\n")
        _f.write(f"seed_arg={args.seed}\n")
        _f.write(f"used_seed={used_seed}\n")
        _f.write(f"out_dir={out_dir}\n")
        _f.write(f"base_dir={base_dir}\n")

    # 加载管线
    controlnet = _load_controlnet(ctrl_dir)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_dir, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_tiling()

    # 复现性生成器
    generator = torch.Generator(device=pipe.device).manual_seed(used_seed)

    # LPIPS（可选）
    lpips_fn = None
    if _has_lpips:
        try:
            lpips_fn = lpips.LPIPS(net='vgg').to(pipe.device).eval()
        except Exception as e:
            print("LPIPS 初始化失败，跳过：", e)
            lpips_fn = None

    # 遍历 CSV 逐张推理并计算指标
    lp_list, ssim_list, msssim_list = [], [], []
    with open(args.csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            src_path = _pick_src(row, args.mode)
            if not src_path:
                continue
            idx = os.path.splitext(os.path.basename(src_path))[0]
            # 将条件图转为 Canny 边缘，并三通道输入（自适应阈值）
            cond_rgb = Image.open(src_path).convert("RGB").resize((SIZE, SIZE))
            cond_np = np.array(cond_rgb)
            gray = cv2.cvtColor(cond_np, cv2.COLOR_RGB2GRAY)
            # 对比度受限自适应直方图均衡（提升微弱边缘）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_eq = clahe.apply(gray)
            # 中值自适应阈值
            med = np.median(gray_eq)
            lo = int(max(0, 0.66 * med))
            hi = int(min(255, 1.33 * med))
            edges = cv2.Canny(gray_eq, lo, hi)
            # 若几乎全黑，回退到固定阈值重试
            if edges.mean() < 1:
                edges = cv2.Canny(gray_eq, 30, 90)
            if edges.mean() < 1:
                edges = cv2.Canny(gray_eq, 10, 30)
            edges_3 = np.stack([edges]*3, axis=2)
            cond = Image.fromarray(edges_3)
            # 保存条件图与灰度图以便检查
            cond_dir = os.path.join(out_dir, "cond")
            os.makedirs(cond_dir, exist_ok=True)
            cond.save(os.path.join(cond_dir, f"{idx}_cond_canny.png"))
            Image.fromarray(gray).save(os.path.join(cond_dir, f"{idx}_gray.png"))
            Image.fromarray(gray_eq).save(os.path.join(cond_dir, f"{idx}_gray_eq.png"))
            img = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                image=cond,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=args.cn_scale,
                generator=generator
            ).images[0]
            # 保存
            suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
            save_path = os.path.join(out_dir, f"{idx}_{suffix}.png")
            img.save(save_path)
            # 计算指标（需要 GT）
            gt_path = row.get("octa_path") if args.mode == "cf2octa" else row.get("cf_path")
            if gt_path and os.path.exists(gt_path):
                lp, s, ms = _compute_metrics(img, Image.open(gt_path).convert("RGB").resize((SIZE, SIZE)), lpips_fn, pipe.device)
                if lp is not None: lp_list.append(lp)
                ssim_list.append(s)
                msssim_list.append(ms)
            if i % 20 == 0:
                print("done", i)

    # 汇总指标
    with open(os.path.join(out_dir, "metrics.txt"), "a") as mf:
        if lp_list:
            mf.write(f"LPIPS_mean={sum(lp_list)/len(lp_list)}\n")
        mf.write(f"SSIM_mean={sum(ssim_list)/len(ssim_list) if ssim_list else 'NA'}\n")
        mf.write(f"MS-SSIM_mean={sum(msssim_list)/len(msssim_list) if msssim_list else 'NA'}\n")
    print("saved to", out_dir)


if __name__ == "__main__":
    main() 