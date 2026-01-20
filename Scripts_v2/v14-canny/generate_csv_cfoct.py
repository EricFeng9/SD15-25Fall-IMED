# -*- coding: utf-8 -*-
"""
CF_OCT 数据集 CSV 生成脚本

【功能】
- 自动扫描 CF_OCT 数据集目录
- 随机划分训练集和测试集（80% / 20%）
- 生成 train_pairs_cfoct.csv 和 test_pairs_cfoct.csv

【使用方法】
python generate_csv_cfoct.py

【数据集结构】
CF_OCT/
  ├── 001/
  │   ├── 001_01.png  # OCT 图
  │   ├── 001_02.png  # CF 图
  │   ├── 001_01.txt  # OCT 关键点
  │   └── 001_02.txt  # CF 关键点
  ├── 002/
  └── ...

【划分策略】
- 随机划分（而非按编号顺序划分）
- 训练集：80% 样本
- 测试集：20% 样本
- 保证训练集和测试集无重叠
"""

import os
import csv
import random

# ============ 配置 ============
DEFAULT_CFOCT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCT_augmented"
OUTPUT_DIR = "/data/student/Fengjunming/SDXL_ControlNet/Scripts"
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train_pairs_cfoct.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test_pairs_cfoct.csv")
RANDOM_SEED = 42  # 固定随机种子，保证可复现


def generate_cfoct_csv(cfoct_root, patient_ids, out_csv):
    """
    生成 CF_OCT 数据集的 CSV 文件
    
    参数:
        cfoct_root: CF_OCT 数据集根目录
        patient_ids: 病例编号列表 (例如: ['001', '002', ...])
        out_csv: 输出 CSV 文件路径
    
    返回:
        成功生成的样本数量
    """
    rows = []
    
    for pid in patient_ids:
        # 处理编号格式
        if isinstance(pid, int):
            pid_str = f"{pid:03d}"
        else:
            pid_str = str(pid)
        
        # 构造文件路径 (01是OCT图，02是CF图)
        patient_dir = os.path.join(cfoct_root, pid_str)
        oct_path = os.path.join(patient_dir, f"{pid_str}_01.png")
        cf_path = os.path.join(patient_dir, f"{pid_str}_02.png")
        oct_pts_path = os.path.join(patient_dir, f"{pid_str}_01.txt")
        cf_pts_path = os.path.join(patient_dir, f"{pid_str}_02.txt")
        
        # 检查文件是否存在
        if not os.path.exists(oct_path):
            print(f"警告: OCT 图像不存在 - {oct_path}")
            continue
        if not os.path.exists(cf_path):
            print(f"警告: CF 图像不存在 - {cf_path}")
            continue
        if not os.path.exists(oct_pts_path):
            print(f"警告: OCT 关键点不存在 - {oct_pts_path}")
            continue
        if not os.path.exists(cf_pts_path):
            print(f"警告: CF 关键点不存在 - {cf_pts_path}")
            continue
        
        rows.append((cf_path, oct_path, cf_pts_path, oct_pts_path))
    
    # 写入 CSV
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cf_path", "oct_path", "cf_pts_path", "oct_pts_path"])
        w.writerows(rows)
    
    print(f"✓ 生成 {out_csv} -> {len(rows)} 个样本")
    return len(rows)


def random_split_train_test(cfoct_root, train_csv, test_csv, train_ratio=0.8, seed=42):
    """
    随机划分训练集和测试集
    
    参数:
        cfoct_root: CF_OCT 数据集根目录
        train_csv: 训练集 CSV 输出路径
        test_csv: 测试集 CSV 输出路径
        train_ratio: 训练集比例 (默认 0.8)
        seed: 随机种子（默认 42，保证可复现）
    
    返回:
        (train_count, test_count)
    """
    # 扫描所有病例文件夹
    all_ids = []
    for item in os.listdir(cfoct_root):
        item_path = os.path.join(cfoct_root, item)
        if os.path.isdir(item_path) and item.isdigit():
            all_ids.append(item)
    
    # 按序号排序（仅用于显示，不影响随机划分）
    all_ids = sorted(all_ids)
    print(f"找到 {len(all_ids)} 个病例: {all_ids[0]} ~ {all_ids[-1]}")
    
    # 设置随机种子并随机打乱
    random.seed(seed)
    shuffled_ids = all_ids.copy()
    random.shuffle(shuffled_ids)
    
    # 随机划分：前80%为训练集，后20%为测试集
    split_idx = int(len(shuffled_ids) * train_ratio)
    train_ids = shuffled_ids[:split_idx]
    test_ids = shuffled_ids[split_idx:]
    
    # 排序后显示（便于查看）
    train_ids_sorted = sorted(train_ids)
    test_ids_sorted = sorted(test_ids)
    
    print(f"\n随机划分结果 (seed={seed}):")
    print(f"训练集: {len(train_ids)} 个病例")
    print(f"  编号: {', '.join(train_ids_sorted)}")
    print(f"测试集: {len(test_ids)} 个病例")
    print(f"  编号: {', '.join(test_ids_sorted)}")
    
    # 生成 CSV
    print("\n正在生成 CSV 文件...")
    train_count = generate_cfoct_csv(cfoct_root, train_ids, train_csv)
    test_count = generate_cfoct_csv(cfoct_root, test_ids, test_csv)
    
    return train_count, test_count


# ============ 主函数 ============
if __name__ == "__main__":
    print("=" * 70)
    print("生成 CF_OCT 数据集 CSV 文件 (随机划分)")
    print("=" * 70)
    print(f"数据集根目录: {DEFAULT_CFOCT_ROOT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"划分比例: 80% 训练集 / 20% 测试集")
    print("=" * 70)
    
    # 随机划分训练集和测试集
    train_count, test_count = random_split_train_test(
        cfoct_root=DEFAULT_CFOCT_ROOT,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        train_ratio=0.8,
        seed=RANDOM_SEED
    )
    
    print("\n" + "=" * 70)
    print(f"✓ 完成！")
    print(f"  训练集: {TRAIN_CSV} ({train_count} 样本)")
    print(f"  测试集: {TEST_CSV} ({test_count} 样本)")
    print(f"  随机种子: {RANDOM_SEED} (可修改以生成不同划分)")
    print("=" * 70)
    
    print("\n【提示】")
    print("- 本脚本使用随机划分策略（而非按编号顺序划分）")
    print("- 固定随机种子保证每次运行结果一致")
    print("- 如需重新划分，可修改脚本中的 RANDOM_SEED 变量")
    print("=" * 70 + "\n")

