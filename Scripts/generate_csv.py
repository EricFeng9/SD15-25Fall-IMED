import os, re, glob, csv
ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCTA_seg"
SEG_TR = os.path.join(ROOT, "seg_trainA")
SEG_TE = os.path.join(ROOT, "seg_testA")
B_TR   = os.path.join(ROOT, "trainB")
B_TE   = os.path.join(ROOT, "testB")

def digits(s):
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def find_by_idx(idx_int, base, key="OCTA"):
    # 尝试常见命名：{idx}OCTA.{ext}、OCTA_{idx}.{ext}
    for ext in (".png",".jpg",".jpeg"):
        p1 = os.path.join(base, f"{idx_int:03d}{key}{ext}")
        if os.path.exists(p1): return p1
        p2 = os.path.join(base, f"{key}_{idx_int:03d}{ext}")
        if os.path.exists(p2): return p2
        p3 = os.path.join(base, f"{idx_int}{key}{ext}")
        if os.path.exists(p3): return p3
        p4 = os.path.join(base, f"{key}_{idx_int}{ext}")
        if os.path.exists(p4): return p4
    # 兜底：目录内按数字索引匹配（忽略前导零）
    for p in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(p):
            continue
        d = digits(os.path.basename(p))
        if d is None:
            continue
        if to_int(d) == idx_int:
            return p
    return None

def index_pairs(seg_dir, b_dir, out_csv):
    rows = []
    for sp in sorted(glob.glob(os.path.join(seg_dir, "*"))):
        if os.path.isdir(sp): continue
        d = digits(os.path.basename(sp))
        if not d: continue
        idx_int = to_int(d)
        if idx_int is None: continue
        bp = find_by_idx(idx_int, b_dir, "OCTA")
        if bp: rows.append((sp, bp))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond_path","target_path"]); w.writerows(rows)
    print(out_csv, "->", len(rows))

index_pairs(SEG_TR, B_TR, "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs.csv")
index_pairs(SEG_TE, B_TE, "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs.csv")