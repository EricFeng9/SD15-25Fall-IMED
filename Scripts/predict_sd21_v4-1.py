'''
推理脚本 - SD 2.1 + HED ControlNet 版本

【版本】v4-sd21
【基于】predict_all_v4.py (SDXL 版本)
【模型】Stable Diffusion 2.1 (768×768) + HED ControlNet
【数据集】配准数据集 (v2-2 格式)

【核心特点】
- 相比 SDXL 推理显存占用降低 60%（4-5GB vs 10-12GB）
- 原生支持 768×768 分辨率
- HED 边缘检测更适合医学图像的软边缘特征
- 支持所有可调参数（prompt, cfg, cn_scale等）

【主要改动】
- Pipeline: StableDiffusionXLControlNetPipeline → StableDiffusionControlNetPipeline
- 移除 original_size 和 target_size 参数（SD 2.1 不需要）
- 简化 Pipeline 调用逻辑

【使用方法】
基础推理:
  python predict_sd21_v4.py --mode cf2octa --name sd21_hed_test --step 8000

高级参数:
  python predict_sd21_v4.py --mode cf2octa --name sd21_hed_test --step 8000 \\
    --prompt "high quality retinal image" \\
    --cn_scale 0.8 --cfg 3.5 --steps 30 --seed 42

参数说明:
  --mode: cf2octa 或 octa2cf
  --name: 训练时的实验名称
  --step: 使用哪个 checkpoint（不填则使用最终模型）
  --ctrl_dir: 直接指定 ControlNet 路径（覆盖 name/step）
  --csv: 推理用的 CSV 文件
  --prompt: 文本提示词（正向）
  --negative_prompt: 负向提示词
  --cn_scale: ControlNet 强度 (0.0-2.0，默认0.8)
  --cfg: Classifier-Free Guidance (1.0-20.0，默认3.5)
  --steps: 去噪步数 (10-100，默认30)
  --seed: 随机种子（用于复现）
'''

import os, csv, torch, argparse
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ============ SD 2.1 模型路径配置 ============
os.environ["HF_HUB_OFFLINE"] = "1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd21"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd21"

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="SD 2.1 + HED ControlNet 推理脚本")

# 基础参数
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                    help="任务方向：cf2octa (CF→OCTA) 或 octa2cf (OCTA→CF)")
parser.add_argument("-n", "--name", dest="name", default="sd21_hed_v4",
                    help="训练时的实验名称")
parser.add_argument("--ctrl_dir", default=None,
                    help="直接指定 ControlNet 权重目录（优先级最高）")
parser.add_argument("--csv", default=csv_path,
                    help="推理使用的 CSV 路径")
parser.add_argument("--step", type=int, default=None,
                    help="选择 step_{N} checkpoint")
parser.add_argument("--savedir", default=None,
                    help="结果保存子目录名")

# 生成参数
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
parser.add_argument("--cn_scale", type=float, default=0.8,
                    help="ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--cfg", type=float, default=3.5,
                    help="Classifier-Free Guidance 强度 (1.0-20.0)")
parser.add_argument("--steps", type=int, default=30,
                    help="去噪步数 (10-100)")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子（用于复现）")

# FP16 支持
parser.add_argument("--use_fp16", action="store_true",
                    help="使用 FP16 推理（降低显存）")

args = parser.parse_args()

# ============ 解析 ControlNet 目录 ============
if args.ctrl_dir:
    ctrl_dir = args.ctrl_dir
else:
    base_ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
    ctrl_dir = os.path.join(base_ctrl_dir, f"step_{args.step}") if args.step else base_ctrl_dir

if not os.path.isdir(ctrl_dir):
    raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

# ============ 输出目录 ============
base_out = os.path.join(out_root, args.mode, args.name)
if args.savedir:
    out_dir = os.path.join(base_out, args.savedir)
else:
    out_dir = os.path.join(base_out, f"step_{args.step}") if args.step else base_out
os.makedirs(out_dir, exist_ok=True)

# ============ 确定随机种子并记录日志 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
log_path = os.path.join(out_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("="*70 + "\n")
    f.write("SD 2.1 + HED ControlNet 推理参数\n")
    f.write("="*70 + "\n")
    f.write(f"mode={args.mode}\n")
    f.write(f"name={args.name}\n")
    f.write(f"ctrl_dir={ctrl_dir}\n")
    f.write(f"csv={args.csv}\n")
    f.write(f"step={args.step}\n")
    f.write(f"savedir={args.savedir}\n")
    f.write(f"prompt={args.prompt}\n")
    f.write(f"negative_prompt={args.negative_prompt}\n")
    f.write(f"cn_scale={args.cn_scale}\n")
    f.write(f"cfg={args.cfg}\n")
    f.write(f"steps={args.steps}\n")
    f.write(f"seed_arg={args.seed}\n")
    f.write(f"used_seed={used_seed}\n")
    f.write(f"use_fp16={args.use_fp16}\n")
    f.write(f"out_dir={out_dir}\n")
    f.write(f"base_dir={base_dir}\n")
    f.write("="*70 + "\n\n")

print("\n" + "="*70)
print("正在加载 SD 2.1 + HED ControlNet 模型...")
print("="*70)
print(f"  Base Model: {base_dir}")
print(f"  ControlNet: {ctrl_dir}")
print(f"  精度: {'FP16' if args.use_fp16 else 'FP32'}")

# ============ 加载模型 ============
dtype = torch.float16 if args.use_fp16 else torch.float32

controlnet = ControlNetModel.from_pretrained(
    ctrl_dir, 
    torch_dtype=dtype, 
    local_files_only=True
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_dir, 
    controlnet=controlnet, 
    torch_dtype=dtype, 
    local_files_only=True
).to("cuda")

# 显存优化
if hasattr(pipe, 'enable_attention_slicing'):
    pipe.enable_attention_slicing("max")
if hasattr(pipe.vae, 'enable_tiling'):
    pipe.vae.enable_tiling()

print("✓ 模型加载完成")
print("="*70 + "\n")

# 生成器（可复现）
generator = torch.Generator(device="cuda").manual_seed(used_seed)

SIZE = 768

def _pick_src(row):
    """根据模式选择源图像路径"""
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    
    if args.mode == "cf2octa":
        return cf or cond
    else:
        return octa or cond

# ============ 推理循环 ============
print("开始推理...")
print(f"  模式: {args.mode}")
print(f"  CSV: {args.csv}")
print(f"  输出: {out_dir}")
print(f"  参数: cn_scale={args.cn_scale}, cfg={args.cfg}, steps={args.steps}\n")

processed_count = 0

with open(args.csv) as f:
    for i, row in enumerate(csv.DictReader(f)):
        src_path = _pick_src(row)
        if not src_path:
            continue
        
        # 加载条件图
        cond = Image.open(src_path).convert("RGB").resize((SIZE, SIZE))
        
        # 保存第一张条件图用于调试
        if processed_count == 0:
            idx = os.path.splitext(os.path.basename(src_path))[0]
            debug_cond_path = os.path.join(out_dir, f"{idx}_input_cond.png")
            cond.save(debug_cond_path)
            print(f"✓ 已保存第一张条件图到: {debug_cond_path}")
            print(f"  源路径: {src_path}\n")
        
        # SD 2.1 推理（简化版，无 original_size/target_size）
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            image=cond,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            controlnet_conditioning_scale=args.cn_scale,
            generator=generator
        ).images[0]
        
        # 保存结果
        idx = os.path.splitext(os.path.basename(src_path))[0]
        suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
        save_path = os.path.join(out_dir, f"{idx}_{suffix}.png")
        img.save(save_path)
        
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"  已处理: {processed_count} 张")

print(f"\n✓ 推理完成！共处理 {processed_count} 张图像")
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}\n")


