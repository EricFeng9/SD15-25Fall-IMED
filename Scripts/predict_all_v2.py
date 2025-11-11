import os, csv, torch, argparse
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

os.environ["HF_HUB_OFFLINE"]="1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sdxl"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2.csv"
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa")  # 任务方向：cf2octa 表示 CF→OCTA，octa2cf 表示 OCTA→CF
parser.add_argument("-n", "--name", "-name", dest="name", default="default")     # 选择训练时的权重名（结果目录会以该名称组织）
parser.add_argument("--ctrl_dir", default=None)                                        # 可选：直接指定 ControlNet 权重目录（优先级最高，覆盖 name/step）
parser.add_argument("--csv", default=csv_path)                                         # 推理使用的 CSV 路径
parser.add_argument("--step", type=int, default=None)                                  # 可选：选择 name 下的 step_{N} 子目录作为权重
parser.add_argument("--savedir", default=None)                                         # 可选：结果保存到 {out_root}/{mode}/{name}/{savedir}，不填则走默认 name/step 规则
# 新增可调参数
parser.add_argument("--prompt", default="")                                           # 文本提示词（正向）
parser.add_argument("--negative_prompt", default="")                                 # 文本提示词（负向），用于抑制不想要的内容
parser.add_argument("--cn_scale", type=float, default=0.8)                             # ControlNet 条件强度（越大越贴输入）
parser.add_argument("--cfg", type=float, default=3.5)                                  # Classifier-Free Guidance（文本引导强度）
parser.add_argument("--steps", type=int, default=30)                                   # 去噪步数（越大越细致、越慢）
parser.add_argument("--seed", type=int, default=None)                                  # 随机种子（填写则推理可复现）
args = parser.parse_args()

# 解析控制网络目录
if args.ctrl_dir:
    ctrl_dir = args.ctrl_dir
else:
    if args.name:
        base_ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
        ctrl_dir = os.path.join(base_ctrl_dir, f"step_{args.step}") if args.step else base_ctrl_dir
    else:
        ctrl_dir = ctrl_root

if not os.path.isdir(ctrl_dir):
    raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

# 输出目录按 mode/name 组织；若指定 --savedir 则使用 name/savedir，否则沿用 name[/step_xxx]
base_out = os.path.join(out_root, args.mode, args.name if args.name else "default")
if args.savedir:
    out_dir = os.path.join(base_out, args.savedir)
else:
    out_dir = os.path.join(base_out, f"step_{args.step}") if args.step else base_out
os.makedirs(out_dir, exist_ok=True)

# 确定本次有效随机种子（即使未显式提供）并写日志
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
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

controlnet = ControlNetModel.from_pretrained(ctrl_dir, torch_dtype=torch.float16, local_files_only=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_dir, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True
).to("cuda")
pipe.enable_attention_slicing("max")
pipe.vae.enable_tiling()

# 生成器（可复现实验）
generator = torch.Generator(device="cuda").manual_seed(used_seed)

SIZE = 768

def _pick_src(row):
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    # v2: 优先用原图列；没有则回退 cond_path
    if args.mode == "cf2octa":
        return cf or cond
    else:
        return octa or cond

with open(args.csv) as f:
    for i, row in enumerate(csv.DictReader(f)):
        src_path = _pick_src(row)
        if not src_path:
            continue
        cond = Image.open(src_path).convert("RGB").resize((SIZE,SIZE))
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            image=cond,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            controlnet_conditioning_scale=args.cn_scale,
            original_size=(SIZE,SIZE),
            target_size=(SIZE,SIZE),
            generator=generator
        ).images[0]
        idx = os.path.splitext(os.path.basename(src_path))[0]
        suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
        img.save(os.path.join(out_dir, f"{idx}_{suffix}.png"))
        if i % 20 == 0:
            print("done", i)
print("saved to", out_dir) 