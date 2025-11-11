import os, re, csv, glob

# 目标：不再使用分割图，直接按数值编号配对 CF 与 OCTA 原图
# 数据根目录（新数据集）
ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCTA_v2"

# 按你提供的目录结构：
# - CF 训练/测试分别在 CF_train / CF_ts
# - OCTA 训练/测试分别在 OCTA_train / OCTA_ts
CF_TRAIN = os.path.join(ROOT, "CF_train")
CF_TEST  = os.path.join(ROOT, "CF_ts")
OCTA_TRAIN = os.path.join(ROOT, "OCTA_train")
OCTA_TEST  = os.path.join(ROOT, "OCTA_ts")

# 输出 CSV 路径维持与 v3 训练/推理脚本一致（仅列名不同）：
TRAIN_OUT = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v3.csv"
TEST_OUT  = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v3.csv"

# 允许的图像扩展名
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _digits(s: str):
    """
    从字符串中提取第一个连续数字串，例如：
    - "139Fundus.png" -> "139"
    - "OCTA_121.png" -> "121"
    无法提取时返回 None。
    """
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _list_images(d: str):
    """
    列出目录 d 下的所有图像文件（不递归）。
    返回绝对路径列表。
    """
    if not os.path.isdir(d):
        return []
    out = []
    # 遍历目录 d 中的所有直接子项（不递归，不进入子目录）
    for p in glob.glob(os.path.join(d, "*")):
        # 如果该路径是目录，则跳过；这里只收集文件
        if os.path.isdir(p):
            continue
        # 提取文件扩展名（如 .png/.jpg），并统一转为小写
        ext = os.path.splitext(p)[1].lower()
        # 仅当扩展名在允许的图片集合 IMG_EXTS 中时，才加入结果列表
        if ext in IMG_EXTS:
            out.append(p)
    return out


def _index_by_id(dir_path: str):
    """
    将目录下的图像按数字编号建立映射：{id_int: filepath}
    - 规则：从文件名中提取第一段数字作为 id。
    - 若同一 id 出现多个文件，保留第一次出现的路径（并可覆盖策略自行调整）。
    """
    mapping = {}
    for p in sorted(_list_images(dir_path)):
        base = os.path.basename(p)
        d = _digits(base)
        if not d:
            continue
        idx = _to_int(d)
        if idx is None:
            continue
        mapping.setdefault(idx, p)
    return mapping


def write_pairs(cf_dir: str, octa_dir: str, out_csv: str):
    """
    将 cf_dir 与 octa_dir 中按相同数字编号的图像进行一一配对，
    输出仅包含两列的 CSV：cf_path, octa_path。
    """
    cf_map = _index_by_id(cf_dir)
    oc_map = _index_by_id(octa_dir)

    common_ids = sorted(set(cf_map.keys()) & set(oc_map.keys()))
    rows = [(cf_map[i], oc_map[i]) for i in common_ids]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cf_path", "octa_path"])  # 仅两列
        w.writerows(rows)

    print(f"生成: {out_csv} -> 共 {len(rows)} 对；CF={len(cf_map)}, OCTA={len(oc_map)}, 交集={len(common_ids)}")
    if rows:
        print("示例样本:")
        for cf_p, oc_p in rows[:3]:
            print(" ", cf_p, "<->", oc_p)


if __name__ == "__main__":
    # 训练集配对：CF_train vs OCTA_train
    write_pairs(CF_TRAIN, OCTA_TRAIN, TRAIN_OUT)
    # 测试集配对：CF_ts vs OCTA_ts
    write_pairs(CF_TEST, OCTA_TEST, TEST_OUT) 