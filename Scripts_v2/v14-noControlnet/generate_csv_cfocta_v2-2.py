"""
生成配准数据集的CSV文件
使用 CF_OCTA_v2_repaired 数据集，包含配准矩阵路径
"""
import os, csv

# v2-2: 使用配准数据集 CF_OCTA_v2_repaired（修正后的命名）
ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCTA_v2_repaired"
CF_TRAIN = os.path.join(ROOT, "CF_train")
CF_TS = os.path.join(ROOT, "CF_ts")
OCTA_TRAIN = os.path.join(ROOT, "OCTA_train")
OCTA_TS = os.path.join(ROOT, "OCTA_ts")
GT_CF_TO_OCTA = os.path.join(ROOT, "GT_CF_to_OCTA")
GT_OCTA_TO_CF = os.path.join(ROOT, "GT_OCTA_to_CF")

def generate_registered_csv(cf_dir, octa_dir, start_idx, end_idx, out_csv):
    """
    生成配准数据集的CSV文件
    
    Args:
        cf_dir: CF图像目录
        octa_dir: OCTA图像目录
        start_idx: 起始编号（包含）
        end_idx: 结束编号（包含）
        out_csv: 输出CSV文件路径
    """
    rows = []
    for idx in range(start_idx, end_idx + 1):
        # 构造文件路径
        cf_path = os.path.join(cf_dir, f"{idx:03d}Fundus.png")
        octa_path = os.path.join(octa_dir, f"{idx:03d}OCTA.png")
        affine_cf_to_octa = os.path.join(GT_CF_TO_OCTA, f"{idx:03d}_CF_to_OCTA_affine.txt")
        affine_octa_to_cf = os.path.join(GT_OCTA_TO_CF, f"{idx:03d}_OCTA_to_CF_affine.txt")
        
        # 检查文件是否存在
        if not os.path.exists(cf_path):
            print(f"警告: CF 图像不存在 - {cf_path}")
            continue
        if not os.path.exists(octa_path):
            print(f"警告: OCTA 图像不存在 - {octa_path}")
            continue
        if not os.path.exists(affine_cf_to_octa):
            print(f"警告: CF->OCTA 配准矩阵不存在 - {affine_cf_to_octa}")
            continue
        if not os.path.exists(affine_octa_to_cf):
            print(f"警告: OCTA->CF 配准矩阵不存在 - {affine_octa_to_cf}")
            continue
        
        rows.append((cf_path, octa_path, affine_cf_to_octa, affine_octa_to_cf))
    
    # 写入CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cf_path", "octa_path", "affine_cf_to_octa_path", "affine_octa_to_cf_path"])
        w.writerows(rows)
    
    print(f"✓ 生成 {out_csv} -> {len(rows)} 个样本")
    return len(rows)

if __name__ == "__main__":
    print("=" * 70)
    print("生成配准数据集 CSV 文件 (v2-2)")
    print("=" * 70)
    
    # 生成训练集 CSV (000-111)
    train_count = generate_registered_csv(
        CF_TRAIN, OCTA_TRAIN, 0, 111,
        "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
    )
    
    # 生成测试集 CSV (112-139)
    test_count = generate_registered_csv(
        CF_TS, OCTA_TS, 112, 139,
        "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
    )
    
    print("\n" + "=" * 70)
    print(f"✓ 完成！训练集: {train_count} 样本, 测试集: {test_count} 样本")
    print("=" * 70) 