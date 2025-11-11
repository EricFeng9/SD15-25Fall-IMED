"""
模型测试脚本 - 对指定目录下的所有图片进行推理

【v10 更新】Single Source of Truth（单一数据源）
- 血管提取函数和参数从 data_loader_all.py 统一导入
- 使用 extract_vessel_with_fov_mask（与训练脚本完全一致）
- 所有 Frangi 参数（gamma, sigmas, alpha, beta）从统一配置获取
- 所有 FOV 掩码参数（erode_pixels, fov_threshold, image_border_margin）从统一配置获取
- 确保推理与训练使用完全相同的血管提取逻辑

【v5 更新】反向resize回原图尺寸 + 结构完全对齐
- 正向：原图 → resize_with_padding到训练集尺寸 → resize到512×512 → 推理
- 反向：512×512输出 → resize到训练集尺寸 → 裁剪padding → resize回原图尺寸
- 记录每个模态的padding信息，确保反向操作精确
- 所有模态（FA、OCT、OCTA）输出原图尺寸，结构完全对齐

【v4 更新】多模态支持 + 训练集尺寸对齐
- 先将输入CF图resize到各模型训练集的CF图原尺寸（使用resize_with_padding）
- CF-FA: 720×576, CF-OCT: 1016×675, CF-OCTA: 400×400
- 然后从对应尺寸提取血管，再resize到512×512推理
- 确保每个模型看到的输入分布与训练时一致
- 支持多模态：cf2fa, cf2oct, cf2octa, all
- all 模式：同时生成 FA、OCT、OCTA 三种模态

【v3 更新】修复预处理顺序不一致问题
- 与训练脚本一致：先从原图提取血管，再 resize 到 512×512
- 确保推理时的数据处理与训练时完全一致，避免血管偏移问题

【v2 更新】智能黑边蒙版
- 使用 gen_mask.py 的 mask_gen 方法自动检测输入图像的黑边区域
- 支持阈值检测和边缘平滑
- 在输出预测图前应用蒙版，保留原图黑边

【功能】
- 读取指定目录下的所有图片
- 使用双路 ControlNet (Scribble + Tile) 进行推理
- 支持单模态或多模态推理
- 智能识别并保留原图的黑边区域（避免黑边区域出现非预期输出）

【处理流程】
1. 读取原始图片（任意尺寸）
2. 【正向】根据模态，resize_with_padding到训练集CF图尺寸（记录padding信息）
   - CF-FA: 720×576, CF-OCT: 1016×675, CF-OCTA: 400×400
3. Resize到512×512
4. 【v10】从512×512图提取血管（使用 data_loader_all.py 统一函数和参数）
5. 模型推理生成 512×512 结果
6. 【反向v5】resize到训练集尺寸 → 裁剪padding → resize回原图尺寸
7. 在原图尺寸应用智能黑边蒙版
8. 输出原图尺寸图像（所有模态结构对齐）

【使用方法】
python model_test.py --name test_experiment --mode cf2fa
python model_test.py --name test_experiment --mode all  # 同时生成三种模态

【输出结构】
输出路径/{name}/
  - {idx}/
    - input_original.png (原图尺寸输入图)
    - pred_fa.png (FA预测图 原图尺寸) [cf2fa模式]
    - pred_oct.png (OCT预测图 原图尺寸) [cf2oct模式]
    - pred_octa.png (OCTA预测图 原图尺寸) [cf2octa模式]
  - log.txt (推理日志)
  
【v10特性】血管提取与训练完全一致（Single Source of Truth）
【v5特性】所有模态输出都是原图尺寸，结构完全对齐
"""

import os
import glob
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel,
                       MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer
from gen_mask import mask_gen  # v2: 智能黑边蒙版生成
from registration_cf_oct import resize_with_padding  # v4: 训练集尺寸对齐

# 【v10 改进】从统一数据加载器导入血管提取函数和参数（Single Source of Truth）
from data_loader_all import (
    extract_vessel_with_fov_mask,  # 统一的血管提取函数
    FRANGI_SIGMAS,                 # Frangi 滤波参数
    FRANGI_ALPHA,                  # Frangi alpha参数
    FRANGI_BETA,                   # Frangi beta参数
    GAMMA_CFFA,                    # CF-FA 数据集 gamma
    GAMMA_CFOCT_CF,                # CF_OCT 数据集 CF 图 gamma
    GAMMA_CFOCTA_CF,               # CF-OCTA 数据集 CF 图 gamma
    get_image_params               # 获取图像处理参数
)

# ============ 配置变量（在程序开头指定）============
# 1. 目标图片路径（待推理的图片目录）
INPUT_IMAGE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_run_test"

# 2. SD15模型和ControlNet模型路径（多模态）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# CF2FA 模型
CONTROLNET_SCRIBBLE_CF2FA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2fa/251031_1/step_8000/controlnet_scribble"
CONTROLNET_TILE_CF2FA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2fa/251031_1/step_8000/controlnet_tile"

# CF2OCT 模型
CONTROLNET_SCRIBBLE_CF2OCT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2oct/251029_3/step_8000/controlnet_scribble"
CONTROLNET_TILE_CF2OCT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2oct/251029_3/step_8000/controlnet_tile"

# CF2OCTA 模型
CONTROLNET_SCRIBBLE_CF2OCTA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2octa/251031_1/step_8000/controlnet_scribble"
CONTROLNET_TILE_CF2OCTA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2octa/251031_1/step_8000/controlnet_tile"

# 3. 输出路径（基础路径）
OUTPUT_BASE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/results/model_test_output"

# ============ 其他配置 ============
SIZE = 512

# ============ 训练集CF图原尺寸（v4新增）============
TRAIN_SIZE_CF2FA = (720, 576)    # CF-FA训练集的CF图尺寸
TRAIN_SIZE_CF2OCT = (1016, 675)  # CF-OCT训练集的CF图尺寸
TRAIN_SIZE_CF2OCTA = (400, 400)  # CF-OCTA训练集的CF图尺寸

# ============ 黑边蒙版参数配置 ============
# 注：这些参数用于最终输出时的黑边蒙版（保留原图黑边区域）
# 与训练时的 fov_threshold 不同（训练时用于血管提取）
MASK_THRESHOLD = 10      # 黑边检测阈值（像素值<threshold视为黑边）
MASK_SMOOTH = True       # 是否平滑蒙版边缘
MASK_KERNEL_SIZE = 5     # 平滑核大小

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="模型测试脚本 v4 - 批量推理（多模态支持）")
parser.add_argument("--name", "-n", required=True,
                    help="实验名称（输出目录名）")
parser.add_argument("--mode", "-m", choices=["cf2fa", "cf2oct", "cf2octa", "all"], required=True,
                    help="推理模式：cf2fa, cf2oct, cf2octa, all（同时生成三种模态）")
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
parser.add_argument("--scribble_scale", type=float, default=0.8,
                    help="Scribble ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--tile_scale", type=float, default=1.0,
                    help="Tile ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--cfg", type=float, default=7.5,
                    help="Classifier-Free Guidance 强度")
parser.add_argument("--steps", type=int, default=30,
                    help="去噪步数 (10-100)")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子（用于复现）")
parser.add_argument("--use_fp16", action="store_true",
                    help="使用 FP16 推理（降低显存）")

args = parser.parse_args()

# ============ 确定推理模式 ============
modes_to_run = []
if args.mode == "all":
    modes_to_run = ["cf2fa", "cf2oct", "cf2octa"]
else:
    modes_to_run = [args.mode]

# ============ 输出目录 ============
out_dir = os.path.join(OUTPUT_BASE_DIR, args.name)
os.makedirs(out_dir, exist_ok=True)

# ============ 检查输入目录 ============
if not os.path.isdir(INPUT_IMAGE_DIR):
    raise FileNotFoundError(f"输入图片目录不存在: {INPUT_IMAGE_DIR}")

# 获取所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, ext)))
    image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, ext.upper())))

image_files = sorted(image_files)  # 按文件名排序

if len(image_files) == 0:
    raise FileNotFoundError(f"在目录 {INPUT_IMAGE_DIR} 中未找到图片文件")

print(f"\n找到 {len(image_files)} 张图片")

# ============ 确定随机种子并记录日志 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
log_path = os.path.join(out_dir, "log.txt")

with open(log_path, "w") as f:
    f.write("="*70 + "\n")
    f.write("模型测试脚本 v5 - 批量推理（反向resize+结构完全对齐）\n")
    f.write("="*70 + "\n")
    f.write(f"实验名称: {args.name}\n")
    f.write(f"推理模式: {args.mode} (运行: {', '.join(modes_to_run)})\n")
    f.write(f"输入目录: {INPUT_IMAGE_DIR}\n")
    f.write(f"输出目录: {out_dir}\n")
    f.write(f"找到图片数: {len(image_files)}\n")
    f.write(f"base_model_dir: {BASE_MODEL_DIR}\n")
    f.write(f"prompt: {args.prompt}\n")
    f.write(f"negative_prompt: {args.negative_prompt}\n")
    f.write(f"scribble_scale: {args.scribble_scale}\n")
    f.write(f"tile_scale: {args.tile_scale}\n")
    f.write(f"cfg: {args.cfg}\n")
    f.write(f"steps: {args.steps}\n")
    f.write(f"seed_arg: {args.seed}\n")
    f.write(f"used_seed: {used_seed}\n")
    f.write(f"use_fp16: {args.use_fp16}\n")
    f.write("【v5更新】反向resize回原图尺寸，所有模态结构完全对齐\n")
    f.write("  正向: 原图 → resize_with_padding(训练集尺寸) → resize(512×512) → 推理\n")
    f.write("  反向: 512×512输出 → resize(训练集尺寸) → 裁剪padding → resize(原图尺寸)\n")
    f.write("【v10更新】Single Source of Truth（单一数据源）:\n")
    f.write("  血管提取函数和参数从 data_loader_all.py 统一导入\n")
    f.write(f"  gamma_cffa: {GAMMA_CFFA}, gamma_cfoct_cf: {GAMMA_CFOCT_CF}, gamma_cfocta_cf: {GAMMA_CFOCTA_CF}\n")
    f.write(f"  frangi_sigmas: {list(FRANGI_SIGMAS)}\n")
    f.write(f"  确保推理与训练使用完全相同的血管提取逻辑\n")
    f.write("【v4更新】训练集尺寸对齐策略:\n")
    f.write(f"  CF-FA训练集尺寸: {TRAIN_SIZE_CF2FA}\n")
    f.write(f"  CF-OCT训练集尺寸: {TRAIN_SIZE_CF2OCT}\n")
    f.write(f"  CF-OCTA训练集尺寸: {TRAIN_SIZE_CF2OCTA}\n")
    f.write("【v2更新】智能黑边蒙版: 使用mask_gen在原图尺寸上检测并保留黑边区域\n")
    f.write(f"  蒙版参数: threshold={MASK_THRESHOLD}, smooth={MASK_SMOOTH}, kernel_size={MASK_KERNEL_SIZE}\n")
    f.write("【架构】双路 ControlNet: Scribble(Vessel) + Tile\n")
    f.write("="*70 + "\n\n")

# ============ 加载模型（支持多模态）============
print("\n" + "="*70)
print(f"正在加载 SD 1.5 + Dual ControlNet 模型...")
print(f"  推理模式: {', '.join(modes_to_run)}")
print("="*70)

os.environ["HF_HUB_OFFLINE"] = "1"
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# 加载共享组件（VAE, UNet, Text Encoder等）
print(f"  Base Model: {BASE_MODEL_DIR}")
vae = AutoencoderKL.from_pretrained(
    BASE_MODEL_DIR, subfolder="vae", torch_dtype=dtype, local_files_only=True
).to(device)
vae.eval()

unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_DIR, subfolder="unet", torch_dtype=dtype, local_files_only=True
).to(device)
unet.eval()

text_encoder = CLIPTextModel.from_pretrained(
    BASE_MODEL_DIR, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    BASE_MODEL_DIR, subfolder="tokenizer", local_files_only=True
)

noise_scheduler = DDPMScheduler.from_pretrained(
    BASE_MODEL_DIR, subfolder="scheduler", local_files_only=True
)
print("✓ SD 1.5 基础模型加载完成")

# 为每个模式加载对应的 ControlNet 和创建 Pipeline
pipelines = {}
model_paths = {
    "cf2fa": (CONTROLNET_SCRIBBLE_CF2FA, CONTROLNET_TILE_CF2FA),
    "cf2oct": (CONTROLNET_SCRIBBLE_CF2OCT, CONTROLNET_TILE_CF2OCT),
    "cf2octa": (CONTROLNET_SCRIBBLE_CF2OCTA, CONTROLNET_TILE_CF2OCTA)
}

for mode in modes_to_run:
    scribble_path, tile_path = model_paths[mode]
    
    print(f"\n  加载 {mode} 模型...")
    print(f"    Scribble: {scribble_path}")
    print(f"    Tile: {tile_path}")
    
    # 加载 ControlNet
    controlnet_scribble = ControlNetModel.from_pretrained(
        scribble_path, 
        torch_dtype=dtype, 
        local_files_only=True
    ).to(device)
    controlnet_scribble.eval()
    
    controlnet_tile = ControlNetModel.from_pretrained(
        tile_path, 
        torch_dtype=dtype, 
        local_files_only=True
    ).to(device)
    controlnet_tile.eval()
    
    # 组合 MultiControlNet
    multi_controlnet = MultiControlNetModel([controlnet_scribble, controlnet_tile])
    
    # 创建 Pipeline
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    # 显存优化
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing("max")
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    
    pipelines[mode] = pipe
    print(f"  ✓ {mode} Pipeline 创建完成")

print("\n✓ 所有模型加载完成")
print("="*70 + "\n")

# ============ 推理循环 ============
print("开始推理...")
print(f"  推理模式: {', '.join(modes_to_run)}")
print(f"  图片数量: {len(image_files)}")
print(f"  输出: {out_dir}")
print(f"  参数: scribble_scale={args.scribble_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}")
print(f"  随机种子: {used_seed}")
print(f"  【v10更新】Single Source of Truth: 血管提取使用 data_loader_all.py 统一函数（与训练一致）")
print(f"  【v5更新】反向resize: 512×512输出 → 训练集尺寸 → 裁剪padding → 原图尺寸（结构完全对齐）")
print(f"  【v4更新】训练集尺寸对齐: CF-FA: {TRAIN_SIZE_CF2FA}, CF-OCT: {TRAIN_SIZE_CF2OCT}, CF-OCTA: {TRAIN_SIZE_CF2OCTA}")
print(f"  【v2更新】智能黑边蒙版: threshold={MASK_THRESHOLD}, smooth={MASK_SMOOTH}, kernel_size={MASK_KERNEL_SIZE}\n")

processed_count = 0

for i, img_path in enumerate(image_files):
    try:
        # 加载原始图像
        src_img_original = Image.open(img_path).convert("RGB")
        
        # 保存原始尺寸
        original_width, original_height = src_img_original.size  # (width, height)
        
        # 【v5新增】生成原图尺寸的黑边蒙版（用于最后应用）
        src_np = np.array(src_img_original)
        mask_original = mask_gen(
            src_np,
            threshold=MASK_THRESHOLD,
            smooth=MASK_SMOOTH,
            kernel_size=MASK_KERNEL_SIZE
        )  # 返回 [0,1] 范围的 float32 数组，黑边为0，其他为1
        
        # 为每张图创建独立文件夹
        idx = os.path.splitext(os.path.basename(img_path))[0]
        img_out_dir = os.path.join(out_dir, idx)
        os.makedirs(img_out_dir, exist_ok=True)
        
        # 保存原图（只保存一次）
        input_original_saved = False
        if not input_original_saved:
            src_img_original.save(os.path.join(img_out_dir, "input_original.png"))
            input_original_saved = True
        
        # 循环遍历每个模式进行推理
        for mode in modes_to_run:
            # 【v4】根据模态选择训练集尺寸
            if mode == "cf2fa":
                train_size = TRAIN_SIZE_CF2FA
            elif mode == "cf2oct":
                train_size = TRAIN_SIZE_CF2OCT
            elif mode == "cf2octa":
                train_size = TRAIN_SIZE_CF2OCTA
            
            # Step 1: resize_with_padding到训练集尺寸（⚠️ 记录padding信息）
            cf_resized_np, pad_top, pad_left, scale_from_func = resize_with_padding(
                np.array(src_img_original),
                target_size=train_size,
                interpolation=cv2.INTER_CUBIC
            )
            cf_resized_pil = Image.fromarray(cf_resized_np)
            
            # 【v5修复】自己计算正确的scale（resize_with_padding返回的可能不准确）
            scale = min(train_size[0] / original_width, train_size[1] / original_height)
            
            # 【v5修复】确保padding信息和有效区域尺寸都是整数
            pad_top = int(pad_top)
            pad_left = int(pad_left)
            h_valid = int(original_height * scale)
            w_valid = int(original_width * scale)
            
            # Step 2: Resize到512×512
            cf_512_pil = cf_resized_pil.resize((SIZE, SIZE))
            
            # Step 3: 【v10 改进】从512×512图提取血管，使用统一配置的参数
            # 根据模式选择对应的 gamma 值（都是 CF 图作为输入）
            if mode == "cf2fa":
                gamma_value = GAMMA_CFFA
            elif mode == "cf2oct":
                gamma_value = GAMMA_CFOCT_CF
            elif mode == "cf2octa":
                gamma_value = GAMMA_CFOCTA_CF
            
            # 获取图像处理参数（用于 FOV 掩码）
            params = get_image_params(mode, param_type='condition')
            
            # 使用统一的血管提取函数（与训练逻辑完全一致）
            cond_scribble = extract_vessel_with_fov_mask(
                cf_512_pil,
                image_type='cf',
                gamma=gamma_value,
                sigmas=FRANGI_SIGMAS,
                alpha=FRANGI_ALPHA,
                beta=FRANGI_BETA,
                apply_fov_mask=params['apply_fov_mask'],
                fov_threshold=params['fov_threshold'],
                erode_pixels=params['erode_pixels'],
                image_border_margin=params['image_border_margin']
            )
            
            # cond_tile直接用512×512的图
            cond_tile = cf_512_pil
            
            # Step 4: 获取对应模式的 pipeline 并推理
            pipe = pipelines[mode]
            generator = torch.Generator(device=device).manual_seed(used_seed)
            
            with torch.no_grad():
                img = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    image=[cond_scribble, cond_tile],
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    controlnet_conditioning_scale=[args.scribble_scale, args.tile_scale],
                    generator=generator
                ).images[0]
            
            # 【v5 反向resize】Step 5-7: 512×512 → 训练集尺寸 → 裁剪padding → 原图尺寸
            
            # Step 5a: 从512×512 resize回训练集尺寸
            pred_train_size = img.resize(train_size)
            pred_train_np = np.array(pred_train_size)
            
            # Step 5b: 裁剪掉padding区域，得到有效内容
            pred_cropped = pred_train_np[pad_top:pad_top+h_valid, pad_left:pad_left+w_valid]
            
            # Step 5c: resize回原图尺寸
            pred_original = Image.fromarray(pred_cropped).resize((original_width, original_height))
            
            # Step 6: 应用原图尺寸的黑边蒙版
            pred_np = np.array(pred_original).astype(np.float32)
            mask_original_3ch = np.stack([mask_original] * 3, axis=2)
            pred_np_masked = pred_np * mask_original_3ch
            pred_np_masked = np.clip(pred_np_masked, 0, 255).astype(np.uint8)
            pred_img_masked = Image.fromarray(pred_np_masked)
            
            # Step 7: 根据模式保存对应的原图尺寸 pred图
            if mode == "cf2fa":
                pred_img_masked.save(os.path.join(img_out_dir, "pred_fa.png"))
            elif mode == "cf2oct":
                pred_img_masked.save(os.path.join(img_out_dir, "pred_oct.png"))
            elif mode == "cf2octa":
                pred_img_masked.save(os.path.join(img_out_dir, "pred_octa.png"))
        
        processed_count += 1
        # 计算黑边区域占比（原图尺寸 mask为0的区域）
        black_pixel_count = np.sum(mask_original == 0)
        total_pixel_count = mask_original.size
        black_ratio = (black_pixel_count / total_pixel_count) * 100
        
        modes_str = ", ".join(modes_to_run)
        print(f"  [{processed_count}/{len(image_files)}] {idx} (原始: {original_width}×{original_height}, 黑边: {black_ratio:.1f}%, 模式: {modes_str}) - 完成")
        
        # 记录到日志
        with open(log_path, "a") as f:
            f.write(f"[{processed_count}] {idx}\n")
            f.write(f"  输入: {img_path}\n")
            f.write(f"  原始尺寸: {original_width}×{original_height}\n")
            f.write(f"  输出尺寸: {original_width}×{original_height} (反向resize回原图尺寸)\n")
            f.write(f"  推理模式: {modes_str}\n")
            f.write(f"  黑边区域占比(原图): {black_ratio:.2f}% ({black_pixel_count}/{total_pixel_count} 像素)\n")
            f.write(f"  蒙版参数: threshold={MASK_THRESHOLD}, smooth={MASK_SMOOTH}, kernel_size={MASK_KERNEL_SIZE}\n")
            f.write(f"  【v10 Single Source of Truth】使用统一配置的血管提取参数:\n")
            if "cf2fa" in modes_to_run:
                f.write(f"    CF-FA: gamma={GAMMA_CFFA}\n")
            if "cf2oct" in modes_to_run:
                f.write(f"    CF-OCT: gamma={GAMMA_CFOCT_CF}\n")
            if "cf2octa" in modes_to_run:
                f.write(f"    CF-OCTA: gamma={GAMMA_CFOCTA_CF}\n")
            f.write(f"  【v5反向resize】处理流程:\n")
            f.write(f"    正向: 输入CF图({original_width}×{original_height}) → resize_with_padding到训练集尺寸 → resize到512×512 → 提取血管(统一配置) → 推理\n")
            f.write(f"    反向: 512×512输出 → resize到训练集尺寸 → 裁剪padding → resize回原图尺寸 → 应用蒙版\n")
            if "cf2fa" in modes_to_run:
                f.write(f"    CF-FA训练集尺寸: {TRAIN_SIZE_CF2FA}\n")
            if "cf2oct" in modes_to_run:
                f.write(f"    CF-OCT训练集尺寸: {TRAIN_SIZE_CF2OCT}\n")
            if "cf2octa" in modes_to_run:
                f.write(f"    CF-OCTA训练集尺寸: {TRAIN_SIZE_CF2OCTA}\n")
            f.write(f"  输出: {img_out_dir}\n")
            f.write("-"*70 + "\n")
    
    except Exception as e:
        print(f"  [{i+1}/{len(image_files)}] {os.path.basename(img_path)} - 失败: {e}")
        with open(log_path, "a") as f:
            f.write(f"[ERROR] {os.path.basename(img_path)}\n")
            f.write(f"  错误: {str(e)}\n")
            f.write("-"*70 + "\n")

# ============ 完成 ============
print(f"\n{'='*70}")
print(f"✓ 推理完成！")
print(f"{'='*70}")
print(f"  共处理: {processed_count} / {len(image_files)} 张图像")
print(f"  架构: 双路 ControlNet (Scribble + Tile) v5")
print(f"  推理模式: {', '.join(modes_to_run)}")
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}")
print(f"  输出结构: 每张图独立文件夹 {{idx}}/")
print(f"    - input_original.png (原图尺寸输入图)")
if "cf2fa" in modes_to_run:
    print(f"    - pred_fa.png (FA预测图 原图尺寸)")
if "cf2oct" in modes_to_run:
    print(f"    - pred_oct.png (OCT预测图 原图尺寸)")
if "cf2octa" in modes_to_run:
    print(f"    - pred_octa.png (OCTA预测图 原图尺寸)")
print(f"  ControlNet 强度: Scribble={args.scribble_scale}, Tile={args.tile_scale}")
print(f"  推理参数: cfg={args.cfg}, steps={args.steps}, seed={used_seed}")
print(f"  【v10更新】Single Source of Truth: 血管提取使用统一配置（与训练完全一致）")
print(f"  【v5更新】反向resize: 512×512输出 → 训练集尺寸 → 裁剪padding → 原图尺寸（结构完全对齐）")
print(f"  【v4更新】训练集尺寸对齐: CF-FA={TRAIN_SIZE_CF2FA}, CF-OCT={TRAIN_SIZE_CF2OCT}, CF-OCTA={TRAIN_SIZE_CF2OCTA}")
print(f"  【v2更新】智能黑边蒙版: threshold={MASK_THRESHOLD}, smooth={MASK_SMOOTH}, kernel_size={MASK_KERNEL_SIZE}")
print(f"{'='*70}\n")

with open(log_path, "a") as f:
    f.write("\n" + "="*70 + "\n")
    f.write("推理完成\n")
    f.write("="*70 + "\n")
    f.write(f"成功处理: {processed_count} / {len(image_files)}\n")
    f.write(f"失败数量: {len(image_files) - processed_count}\n")
    f.write(f"推理模式: {', '.join(modes_to_run)}\n")
    f.write(f"输出尺寸: 原图尺寸 (反向resize+应用智能蒙版)\n")
    f.write(f"【v10更新】Single Source of Truth: 血管提取使用统一配置（与训练完全一致）\n")
    f.write(f"  gamma_cffa={GAMMA_CFFA}, gamma_cfoct_cf={GAMMA_CFOCT_CF}, gamma_cfocta_cf={GAMMA_CFOCTA_CF}\n")
    f.write(f"【v5更新】所有模态输出结构完全对齐（都回到原图尺寸）\n")
    f.write(f"训练集尺寸对齐: CF-FA={TRAIN_SIZE_CF2FA}, CF-OCT={TRAIN_SIZE_CF2OCT}, CF-OCTA={TRAIN_SIZE_CF2OCTA}\n")
    f.write("="*70 + "\n")

