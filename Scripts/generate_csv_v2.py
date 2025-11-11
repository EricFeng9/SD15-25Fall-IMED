import os, re, glob, csv
ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCTA_seg"
# 假定原图目录与分割目录平行，名称去掉 seg_ 前缀即可得到原图目录
SEG_TR = os.path.join(ROOT, "seg_trainA")
SEG_TE = os.path.join(ROOT, "seg_testA")
A_TR   = os.path.join(ROOT, "trainA")   # CF 原图 trainA
A_TE   = os.path.join(ROOT, "testA")    # CF 原图 testA
B_TR   = os.path.join(ROOT, "trainB")   # OCTA 原图 trainB
B_TE   = os.path.join(ROOT, "testB")    # OCTA 原图 testB

# 既支持 00OCTA.png，也支持 OCTA_00.png 等

def digits(s):
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def find_by_idx(idx_int, base, key="OCTA"):
    for ext in (".png",".jpg",".jpeg"):
        p1 = os.path.join(base, f"{idx_int:03d}{key}{ext}")
        if os.path.exists(p1): return p1
        p2 = os.path.join(base, f"{key}_{idx_int:03d}{ext}")
        if os.path.exists(p2): return p2
        p3 = os.path.join(base, f"{idx_int}{key}{ext}")
        if os.path.exists(p3): return p3
        p4 = os.path.join(base, f"{key}_{idx_int}{ext}")
        if os.path.exists(p4): return p4
    for p in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(p):
            continue
        d = digits(os.path.basename(p))
        if d is None:
            continue
        if to_int(d) == idx_int:
            return p
    return None

# 从 segA 找 A 原图：同名数字索引，去对应的 A 目录里找 CF 原图
# 从 segA 找 B 原图：同名数字索引，在 B 目录里找 OCTA

def index_pairs(seg_dir, a_dir, b_dir, out_csv):
    rows = []
    for sp in sorted(glob.glob(os.path.join(seg_dir, "*"))):
        if os.path.isdir(sp): continue
        d = digits(os.path.basename(sp))
        if not d: continue
        idx_int = to_int(d)
        if idx_int is None: continue
        cf_orig = find_by_idx(idx_int, a_dir, key="FUNDUS") or find_by_idx(idx_int, a_dir, key="CF")
        octa_orig = find_by_idx(idx_int, b_dir, key="OCTA")
        if not octa_orig:
            continue
        # 允许 cf_orig 缺失，训练/推理 v2 会兜底从 cond_path 推断
        rows.append((cf_orig or "", octa_orig, sp, octa_orig))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cf_path","octa_path","cond_path","target_path"])
        w.writerows(rows)
    print(out_csv, "->", len(rows))

index_pairs(SEG_TR, A_TR, B_TR, "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2.csv")
index_pairs(SEG_TE, A_TE, B_TE, "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2.csv") 