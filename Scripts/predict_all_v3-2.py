import os, csv, argparse
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Frangi 依赖
import numpy as np
try:
    from skimage.filters import frangi
    from skimage import exposure
    _has_skimage = True
except Exception:
    _has_skimage = False

os.environ["HF_HUB_OFFLINE"] = "1"

base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd21-diffusers"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v3.csv"
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd21_v3_2"
SIZE = 768


def _load_controlnet(ctrl_dir: str):
    return ControlNetModel.from_pretrained(ctrl_dir, torch_dtype=torch.float16, local_files_only=True)


def _pick_src(row: dict, mode: str) -> str:
    if mode == "cf2octa":
        return row.get("cf_path")
    else:
        return row.get("octa_path")

# ---------- Frangi 条件生成 ----------
# 返回规范化到 [0,1] 的 Frangi 灰度二维数组（numpy）
def _frangi_gray_from_pil(src_pil: Image.Image) -> np.ndarray:
    if not _has_skimage:
        raise ImportError("本脚本需要 scikit-image，请先安装：pip install scikit-image")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
    parser.add_argument("--name", default="v3_2_frangi")
    parser.add_argument("--ctrl_dir", default=None)
    parser.add_argument("--csv", default=csv_path if 'csv_path' in globals() else csv_path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--savedir", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--cn_scale", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=6.5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.ctrl_dir:
        ctrl_dir = args.ctrl_dir
    else:
        base_ctrl = os.path.join(ctrl_root, args.mode, args.name if args.name else "v3_2_frangi")
        ctrl_dir = os.path.join(base_ctrl, f"step_{args.step}") if args.step else base_ctrl
    if not os.path.isdir(ctrl_dir):
        raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

    base_out = os.path.join(out_root, args.mode, args.name if args.name else "v3_2_frangi")
    out_dir = os.path.join(base_out, args.savedir) if args.savedir else (os.path.join(base_out, f"step_{args.step}") if args.step else base_out)
    os.makedirs(out_dir, exist_ok=True)

    used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))

    with open(os.path.join(out_dir, "log.txt"), "a") as _f:
        _f.write("inference_params\n"); _f.write(f"mode={args.mode}\n"); _f.write(f"name={args.name}\n")
        _f.write(f"ctrl_dir={ctrl_dir}\n"); _f.write(f"csv={args.csv}\n"); _f.write(f"savedir={args.savedir}\n")
        _f.write(f"prompt={args.prompt}\n"); _f.write(f"neg={args.negative_prompt}\n")
        _f.write(f"cn_scale={args.cn_scale}\n"); _f.write(f"cfg={args.cfg}\n"); _f.write(f"steps={args.steps}\n")
        _f.write(f"seed_arg={args.seed}\n"); _f.write(f"used_seed={used_seed}\n")

    controlnet = _load_controlnet(ctrl_dir)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_dir, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing("max"); pipe.vae.enable_tiling()

    generator = torch.Generator(device=pipe.device).manual_seed(used_seed)

    cond_dir = os.path.join(out_dir, "cond"); os.makedirs(cond_dir, exist_ok=True)

    with open(args.csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            src_path = _pick_src(row, args.mode)
            if not src_path: continue
            idx = os.path.splitext(os.path.basename(src_path))[0]
            src_pil = Image.open(src_path).convert("RGB").resize((SIZE,SIZE))
            # Frangi 条件（灰度与 RGB）
            fr_gray = _frangi_gray_from_pil(src_pil)
            fr_rgb = _rgb_from_gray(fr_gray)
            # 保存条件图
            fr_rgb.save(os.path.join(cond_dir, f"{idx}_cond_frangi.png"))
            Image.fromarray((fr_gray * 255.0).astype(np.uint8), mode="L").save(os.path.join(cond_dir, f"{idx}_cond_frangi_gray.png"))
            img = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                image=fr_rgb,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=args.cn_scale,
                generator=generator
            ).images[0]
            suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
            img.save(os.path.join(out_dir, f"{idx}_{suffix}.png"))
            if i % 20 == 0: print("done", i)
    print("saved to", out_dir)

if __name__ == "__main__":
    main() 