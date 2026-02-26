# -*- coding: utf-8 -*-
"""
RefineNet 训练脚本 - Stage 2 风格增强模块

训练流程:
1. 加载训练好的 CF2FA 模型（冻结）
2. 对数据集中的每个 CF 生成 fa_gen
3. 使用 RefineNet 学习: [cf, fa_gen] -> fa_refined，使其接近 fa_gt
4. 只训练 RefineNet，CF2FA 完全冻结

Loss: L1 + Texture + Intensity (+ 可选 Perceptual)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# 导入 RefineNet 和损失函数
from refine_net import RefineNet
from loss_utils import RefineNetLoss

# 导入 Diffusers 组件（用于加载 CF2FA 模型）
from diffusers import (
    DDPMScheduler, ControlNetModel, AutoencoderKL, 
    UNet2DConditionModel, StableDiffusionControlNetPipeline,
    MultiControlNetModel
)
from transformers import CLIPTextModel, CLIPTokenizer

# 导入数据集
sys.path.append(os.path.join(os.path.dirname(__file__), 
                             "../../data/operation_pre_filtered_cffa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset

# 导入血管检测器（用于 CF2FA 生成）
sys.path.append(os.path.join(os.path.dirname(__file__), "../v16"))
from vessle_detector import extract_vessel_map

# ============ 全局配置 ============
DEVICE = torch.device("cuda")
SIZE = 512
experiment_name = '260128_2'
# CF2FA 模型路径配置
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
CF2FA_CHECKPOINT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/{experiment_name}/260121_2/best_checkpoint"

# 输出路径
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/style_transfer_refine_v1"

# ============ 辅助函数 ============

def load_cf2fa_pipeline(checkpoint_dir, device):
    """
    加载训练好的 CF2FA 模型（双路 ControlNet + SD1.5）
    返回一个可直接用于推理的 pipeline，所有组件冻结
    """
    print(f"正在加载 CF2FA 模型: {checkpoint_dir}")
    
    # 加载基础模型
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(device)
    
    # 加载 ControlNet
    scribble_path = os.path.join(checkpoint_dir, "controlnet_scribble")
    tile_path = os.path.join(checkpoint_dir, "controlnet_tile")
    
    cn_scribble = ControlNetModel.from_pretrained(scribble_path).to(device)
    cn_tile = ControlNetModel.from_pretrained(tile_path).to(device)
    
    # 冻结所有参数
    for model in [text_encoder, vae, unet, cn_scribble, cn_tile]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    # 构建 pipeline
    multi_controlnet = MultiControlNetModel([cn_scribble, cn_tile])
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(device)
    
    pipe.set_progress_bar_config(disable=True)
    
    print("✓ CF2FA 模型加载完成\n")
    return pipe


@torch.no_grad()
def generate_fa_from_cf(cf_tensor, pipeline, scribble_scale=0.8, tile_scale=1.0, 
                        num_steps=25, seed=42):
    """
    使用 CF2FA pipeline 生成 FA 图像
    
    Args:
        cf_tensor: (1, 3, H, W) 或 (3, H, W)，范围 [0, 1]，RGB 格式
        pipeline: CF2FA StableDiffusionControlNetPipeline
        scribble_scale, tile_scale: ControlNet 条件强度
        num_steps: 去噪步数
        seed: 随机种子
    
    Returns:
        fa_gen: (1, 3, H, W)，范围 [0, 1]
    """
    if cf_tensor.dim() == 3:
        cf_tensor = cf_tensor.unsqueeze(0)
    
    cf_tensor = cf_tensor.to(DEVICE)
    
    # 提取血管图作为 scribble 条件
    vessel_map = extract_vessel_map(cf_tensor, image_type='cf', mode='cf2fa')
    cond_scribble = vessel_map.repeat(1, 3, 1, 1)  # (1, 3, H, W)
    cond_tile = cf_tensor  # (1, 3, H, W)
    
    # 推理
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    h, w = cf_tensor.shape[2], cf_tensor.shape[3]
    
    output_img = pipeline(
        prompt="",
        image=[cond_scribble, cond_tile],
        num_inference_steps=num_steps,
        controlnet_conditioning_scale=[scribble_scale, tile_scale],
        generator=generator,
        width=w,
        height=h
    ).images[0]
    
    # PIL -> Tensor [0, 1]
    from torchvision import transforms
    fa_gen = transforms.ToTensor()(output_img).unsqueeze(0).to(DEVICE)
    
    return fa_gen


def save_comparison(cf, fa_gen, fa_refined, fa_gt, save_path):
    """保存对比图：CF | FA_gen | FA_refined | FA_gt"""
    import matplotlib.pyplot as plt
    
    # 转换为 numpy，范围 [0, 1]
    cf_np = cf[0].cpu().permute(1, 2, 0).numpy()
    fa_gen_np = fa_gen[0].cpu().permute(1, 2, 0).numpy()
    fa_refined_np = fa_refined[0].cpu().permute(1, 2, 0).numpy()
    fa_gt_np = (fa_gt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2  # [-1,1] -> [0,1]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cf_np)
    axes[0].set_title('CF (Input)')
    axes[0].axis('off')
    
    axes[1].imshow(fa_gen_np)
    axes[1].set_title('FA Gen (Stage 1)')
    axes[1].axis('off')
    
    axes[2].imshow(fa_refined_np)
    axes[2].set_title('FA Refined (Stage 2)')
    axes[2].axis('off')
    
    axes[3].imshow(fa_gt_np)
    axes[3].set_title('FA GT')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============ 主训练流程 ============

def main():
    parser = argparse.ArgumentParser(description="RefineNet Stage 2 训练")
    parser.add_argument("-n", "--name", default="exp_refine_v1", help="实验名称")
    parser.add_argument("--cf2fa_checkpoint", default=CF2FA_CHECKPOINT, help="CF2FA 模型路径")
    parser.add_argument("--max_epochs", type=int, default=50, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--base_ch", type=int, default=32, help="RefineNet 基础通道数")
    
    # Loss 权重
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_texture", type=float, default=0.2)
    parser.add_argument("--lambda_intensity", type=float, default=0.05)
    parser.add_argument("--lambda_perceptual", type=float, default=0.0, help="感知损失权重（0=不启用）")
    
    # CF2FA 推理参数
    parser.add_argument("--scribble_scale", type=float, default=0.8)
    parser.add_argument("--tile_scale", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=25, help="CF2FA 去噪步数")
    
    args = parser.parse_args()
    
    # 创建输出目录
    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # 1. 加载 CF2FA 模型（冻结）
    cf2fa_pipe = load_cf2fa_pipeline(args.cf2fa_checkpoint, DEVICE)
    
    # 2. 加载数据集
    print("正在加载数据集...")
    train_dataset = CFFADataset(split='train', mode='cf2fa')
    val_dataset = CFFADataset(split='test', mode='cf2fa')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, 
                           shuffle=False, num_workers=2)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本\n")
    
    # 3. 初始化 RefineNet
    print(f"正在初始化 RefineNet (base_ch={args.base_ch})...")
    refine_net = RefineNet(base_ch=args.base_ch).to(DEVICE)
    
    total_params = sum(p.numel() for p in refine_net.parameters() if p.requires_grad)
    print(f"RefineNet 可训练参数: {total_params / 1e6:.2f}M\n")
    
    # 4. 初始化损失函数和优化器
    criterion = RefineNetLoss(
        lambda_l1=args.lambda_l1,
        lambda_texture=args.lambda_texture,
        lambda_intensity=args.lambda_intensity,
        lambda_perceptual=args.lambda_perceptual,
        device=DEVICE
    )
    
    optimizer = torch.optim.AdamW(refine_net.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )
    
    # 5. 训练循环
    print(f"开始训练 - 最大轮数: {args.max_epochs}\n")
    best_val_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        refine_net.train()
        epoch_losses = []
        epoch_start = time.time()
        
        for batch_idx, (cf, fa_gt, _, _) in enumerate(train_loader):
            cf = cf.to(DEVICE)       # (B, 3, 512, 512), [0, 1]
            fa_gt = fa_gt.to(DEVICE) # (B, 3, 512, 512), [-1, 1]
            
            # Step 1: 用 CF2FA 生成 fa_gen（冻结，无梯度）
            with torch.no_grad():
                # 批量处理：对每个样本单独生成
                fa_gen_list = []
                for i in range(cf.shape[0]):
                    fa_gen_single = generate_fa_from_cf(
                        cf[i:i+1], cf2fa_pipe, 
                        args.scribble_scale, args.tile_scale, 
                        args.num_steps
                    )
                    fa_gen_list.append(fa_gen_single)
                fa_gen = torch.cat(fa_gen_list, dim=0)  # (B, 3, 512, 512), [0, 1]
            
            # Step 2: RefineNet 前向：[cf, fa_gen] -> fa_refined
            cf_gray = cf.mean(dim=1, keepdim=True)  # (B, 1, 512, 512)
            fa_refined = refine_net(cf_gray, fa_gen)  # (B, 3, 512, 512), [0, 1]
            
            # Step 3: 计算损失（fa_refined vs fa_gt）
            # 需要把 fa_gt 从 [-1, 1] 转到 [0, 1]
            fa_gt_01 = (fa_gt + 1) / 2
            
            loss, loss_dict = criterion(fa_refined, fa_gt_01)
            
            # Step 4: 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refine_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict['total'])
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.max_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] | "
                      f"Loss: {loss_dict['total']:.4f} | "
                      f"L1: {loss_dict['l1']:.4f} | "
                      f"Tex: {loss_dict['texture']:.4f} | "
                      f"Int: {loss_dict['intensity']:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # Epoch 统计
        avg_train_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        
        print(f"\n[Epoch {epoch+1}] 训练完成 | "
              f"平均Loss: {avg_train_loss:.4f} | "
              f"用时: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")
        
        # ============ 每轮都进行验证 ============
        refine_net.eval()
        val_losses = []
        
        # 用于保存可视化的样本数据（前3个）
        vis_samples = []
        
        print("正在验证...")
        with torch.no_grad():
            for val_idx, (cf, fa_gt, cf_path, _) in enumerate(val_loader):
                if val_idx >= 10:  # 只验证前 10 个样本
                    break
                
                cf = cf.to(DEVICE)
                fa_gt = fa_gt.to(DEVICE)
                
                # 生成 fa_gen
                fa_gen = generate_fa_from_cf(cf, cf2fa_pipe, 
                                             args.scribble_scale, args.tile_scale,
                                             args.num_steps)
                
                # RefineNet 推理
                cf_gray = cf.mean(dim=1, keepdim=True)
                fa_refined = refine_net(cf_gray, fa_gen)
                
                # 计算验证损失
                fa_gt_01 = (fa_gt + 1) / 2
                loss, loss_dict = criterion(fa_refined, fa_gt_01)
                val_losses.append(loss_dict['total'])
                
                # 保存前 3 个样本的数据，用于可能的可视化
                if val_idx < 3:
                    vis_samples.append({
                        'cf': cf,
                        'fa_gen': fa_gen,
                        'fa_refined': fa_refined,
                        'fa_gt': fa_gt,
                        'cf_path': cf_path
                    })
        
        avg_val_loss = np.mean(val_losses)
        print(f"验证Loss: {avg_val_loss:.4f}\n")
        
        # ============ 判断是否需要可视化 ============
        is_best = avg_val_loss < best_val_loss
        is_milestone = (epoch + 1) % 5 == 0 or (epoch + 1) % 10 == 0
        should_visualize = is_best or is_milestone
        
        # 如果需要可视化，保存前3个样本
        if should_visualize:
            print(f"保存可视化 (原因: {'刷新Best' if is_best else f'Epoch {epoch+1} 里程碑'})...")
            for sample in vis_samples:
                save_name = os.path.basename(sample['cf_path'][0]).replace('.png', 
                                                                           f'_epoch{epoch+1}.png')
                save_comparison(sample['cf'], sample['fa_gen'], sample['fa_refined'], 
                               sample['fa_gt'], os.path.join(vis_dir, save_name))
            print(f"✓ 可视化已保存\n")
        
        # ============ 保存最佳模型 ============
        if is_best:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': refine_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (Epoch {epoch+1}, Val Loss: {best_val_loss:.4f})\n")
        
        # ============ 定期保存 checkpoint ============
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': refine_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(ckpt_dir, f'checkpoint_epoch{epoch+1}.pth'))
            print(f"✓ 保存 Checkpoint (Epoch {epoch+1})\n")
        
        # 记录日志
        with open(os.path.join(out_dir, "training_log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                   f"Best: {best_val_loss:.4f}\n")
    
    print("\n训练完成！")
    print(f"最佳验证Loss: {best_val_loss:.4f}")
    print(f"结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
