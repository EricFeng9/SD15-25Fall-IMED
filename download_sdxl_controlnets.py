#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 SDXL ControlNet 模型脚本
"""

import os
from huggingface_hub import snapshot_download

# 目标目录
MODELS_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models"

# SDXL ControlNet 模型列表
models = {
    # Scribble/Sketch ControlNet for SDXL
    # 注意：SDXL 官方可能没有专门的 scribble 模型，可以使用 canny 或 scribble-hed 替代
    "controlnet-sdxl-scribble": {
        "repo_id": "xinsir/controlnet-scribble-sdxl-1.0",  # 使用社区版本
        "description": "SDXL Scribble ControlNet (用于边缘/草图控制)"
    },
    # Tile ControlNet for SDXL
    "controlnet-sdxl-tile": {
        "repo_id": "xinsir/controlnet-tile-sdxl-1.0",
        "description": "SDXL Tile ControlNet (用于图像细节增强)"
    },
}

# 备选方案：如果上面的 scribble 不行，可以尝试这些
alternative_models = {
    # Canny 可以作为 Scribble 的替代
    "controlnet-sdxl-canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "description": "SDXL Canny ControlNet (边缘检测，可替代 Scribble)"
    },
    # 或者使用 OpenPose
    "controlnet-sdxl-openpose": {
        "repo_id": "thibaud/controlnet-openpose-sdxl-1.0",
        "description": "SDXL OpenPose ControlNet"
    },
}

def download_model(model_name, repo_id, description):
    """下载单个模型"""
    save_path = os.path.join(MODELS_DIR, model_name)
    
    print(f"\n{'='*60}")
    print(f"正在下载: {model_name}")
    print(f"描述: {description}")
    print(f"仓库: {repo_id}")
    print(f"保存路径: {save_path}")
    print(f"{'='*60}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ {model_name} 下载完成！")
        return True
    except Exception as e:
        print(f"❌ {model_name} 下载失败: {e}")
        print(f"   尝试访问: https://huggingface.co/{repo_id}")
        return False

def main():
    print("开始下载 SDXL ControlNet 模型...")
    print(f"目标目录: {MODELS_DIR}\n")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    success_count = 0
    failed_models = []
    
    # 先尝试主要模型
    for model_name, info in models.items():
        if download_model(model_name, info["repo_id"], info["description"]):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # 如果 scribble 下载失败，提示使用备选方案
    if "controlnet-sdxl-scribble" in failed_models:
        print(f"\n{'='*60}")
        print("⚠️  SDXL Scribble ControlNet 下载失败")
        print("💡 建议使用以下备选方案：")
        print("   1. Canny ControlNet (边缘检测，效果类似)")
        print("   2. 使用 SD1.5 的 Scribble ControlNet")
        print(f"{'='*60}\n")
        
        # 询问是否下载 Canny 作为替代
        print("是否下载 Canny ControlNet 作为替代？")
        print("提示：Canny 边缘检测可以很好地替代 Scribble 功能")
        
        # 自动下载 Canny 作为备选
        print("\n自动下载 Canny ControlNet 作为备选...")
        if download_model(
            "controlnet-sdxl-canny",
            alternative_models["controlnet-sdxl-canny"]["repo_id"],
            alternative_models["controlnet-sdxl-canny"]["description"]
        ):
            print("\n✅ 已下载 Canny ControlNet，可以在代码中使用它替代 Scribble")
            print("   修改方法：将 SCRIBBLE_CN_DIR 指向 controlnet-sdxl-canny")
    
    print(f"\n{'='*60}")
    print(f"下载完成！成功: {success_count}/{len(models)}")
    if failed_models:
        print(f"失败的模型: {', '.join(failed_models)}")
    print(f"{'='*60}")
    
    # 列出下载的模型
    print("\n已下载的 SDXL ControlNet 模型:")
    for model_name in list(models.keys()) + ["controlnet-sdxl-canny"]:
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            print(f"  ✓ {model_name}")
        else:
            print(f"  ✗ {model_name} (未找到)")
    
    # 打印使用说明
    print(f"\n{'='*60}")
    print("📝 使用说明：")
    print("1. 如果下载了 Canny ControlNet，需要修改训练脚本：")
    print("   SCRIBBLE_CN_DIR = '.../models/controlnet-sdxl-canny'")
    print("\n2. 或者创建软链接：")
    print("   cd /data/student/Fengjunming/SDXL_ControlNet/models")
    print("   ln -s controlnet-sdxl-canny controlnet-sdxl-scribble")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
