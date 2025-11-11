import os, csv, torch, argparse
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

os.environ["HF_HUB_OFFLINE"]="1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
# 默认权重根目录，与训练脚本保持一致
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sdxl"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs.csv"
# 输出目录根，后续按 mode/name 组织
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")
# 支持传入名称，优先从 root/mode/name 读取；若未提供则使用 --ctrl_dir
parser.add_argument("--n", "--name", dest="name", default="default")
parser.add_argument("--ctrl_dir", default=None)
args = parser.parse_args()

# 解析控制网络目录
if args.name:
    ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
else:
    # 向后兼容：如果未给 name，则回退到提供的 ctrl_dir 或默认 root
    ctrl_dir = args.ctrl_dir or ctrl_root

# 输出目录按 mode/name 组织
out_dir = os.path.join(out_root, args.mode, args.name)
os.makedirs(out_dir, exist_ok=True)

controlnet = ControlNetModel.from_pretrained(ctrl_dir, torch_dtype=torch.float16, local_files_only=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_dir, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True
).to("cuda")
pipe.enable_attention_slicing("max")
pipe.vae.enable_tiling()

def _pick_paths(row):
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt  = row.get("target_path")
    if cf and octa:
        return (cf, octa) if args.mode == "cf2octa" else (octa, cf)
    return (cond, tgt) if args.mode == "cf2octa" else (tgt, cond)

with open(csv_path) as f:
    for i, row in enumerate(csv.DictReader(f)):
        src_path, _ = _pick_paths(row)
        cond = Image.open(src_path).convert("RGB").resize((768,768))
        img = pipe(
            prompt="",
            image=cond,
            num_inference_steps=30,
            guidance_scale=3.5,
            controlnet_conditioning_scale=0.8,
            original_size=(768,768),
            target_size=(768,768)
        ).images[0]
        idx = os.path.splitext(os.path.basename(src_path))[0]
        suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
        img.save(os.path.join(out_dir, f"{idx}_{suffix}.png"))
        if i % 20 == 0:
            print("done", i)
print("saved to", out_dir)