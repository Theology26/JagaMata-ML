"""
=============================================================================
JagaMata ML Training Suite - Common Utilities & Data Engine
=============================================================================
Menyediakan helper untuk:
1. Penemuan otomatis direktori dataset (AFLW2000_300W-LP, Myopia images, ml/datasets)
2. Pembaca label pose .mat (pitch, yaw, roll) baik via scipy maupun pure-Python parser
3. Ekstraksi group_id_from_name anti-leakage untuk 300W-LP
4. Group-based split (300W-LP) dengan verifikasi ketat irisan group = 0
5. Stratified split (Myopia fundus) 70/15/15 per kelas
6. Diagnostic hardware & optimasi lingkungan (RTX 5050, i7-13620H, RAM 16GB)
7. Kalkulasi metrik: MAE, Bootstrap 95% CI (1000 resamples), ROC-AUC, Sensitivity, Specificity, F1
=============================================================================
"""

import os
import re
import math
import zlib
import json
import time
import random
import struct
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# 1. PENCARIAN DIREKTORI DATASET (FLEXIBLE DISCOVERY)
# =============================================================================

def find_first_existing(candidates: List[Path or str]) -> Optional[Path]:
    """Mencari path pertama yang benar-benar ada di sistem file."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p.resolve()
    return None

def get_300wlp_root() -> Path:
    """
    Mendapatkan root direktori 300W-LP yang berisi subfolder AFW, HELEN, IBUG, LFPW.
    Mendukung folder AFLW2000_300W-LP di root workspace maupun di ml/datasets/.
    """
    candidates = [
        Path("AFLW2000_300W-LP/300W-LP/300W_LP"),
        Path("AFLW2000_300W-LP/300W_LP"),
        Path("ml/datasets/aflw2000-300wlp/300W_LP"),
        Path("ml/datasets/aflw2000-300wlp/300W-LP/300W_LP"),
        Path("300W-LP/300W_LP"),
        Path("300W_LP"),
    ]
    p = find_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            f"Direktori 300W-LP tidak ditemukan! Kandidat pencarian: {[str(c) for c in candidates]}"
        )
    return p

def get_aflw2000_root() -> Path:
    """
    Mendapatkan root direktori AFLW2000-3D untuk evaluasi test final (SEKALI SAJA di akhir).
    """
    candidates = [
        Path("AFLW2000_300W-LP/AFLW2000-3D/AFLW2000"),
        Path("AFLW2000_300W-LP/AFLW2000"),
        Path("ml/datasets/aflw2000-3d/AFLW2000"),
        Path("AFLW2000-3D/AFLW2000"),
        Path("AFLW2000"),
    ]
    p = find_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            f"Direktori AFLW2000-3D tidak ditemukan! Kandidat pencarian: {[str(c) for c in candidates]}"
        )
    return p

def get_myopia_root() -> Path:
    """
    Mendapatkan root direktori Myopia images yang berisi Myopia_images dan Normal_images.
    Mendukung 'Myopia images', 'Myiopia Images', dan 'ml/datasets/myopia-images'.
    """
    candidates = [
        Path("Myopia images/Images"),
        Path("Myopia images"),
        Path("Myiopia Images/Images"),
        Path("Myiopia Images"),
        Path("ml/datasets/myopia-images/Images"),
        Path("ml/datasets/myopia-images"),
    ]
    p = find_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            f"Direktori Myopia Images tidak ditemukan! Kandidat pencarian: {[str(c) for c in candidates]}"
        )
    if (p / "Images").exists():
        return (p / "Images").resolve()
    return p

# =============================================================================
# 2. PEMBACA LABEL POSE .MAT (ROBUST POSE READER)
# =============================================================================

def _parse_mat_pure_python(mat_path: Path or str) -> Tuple[float, float, float]:
    """
    Parser biner darurat MAT 5.0 murni Python (tanpa dependency eksternal).
    Membaca Pose_Para (pitch, yaw, roll) berukuran 7 floats/doubles dan mengembalikan derajat.
    """
    with open(mat_path, "rb") as f:
        _ = f.read(128)  # header
        data = f.read()

    offset = 0
    while offset < len(data):
        if offset + 8 > len(data):
            break
        dtype, size = struct.unpack("<II", data[offset:offset+8])
        if dtype == 15:  # miCOMPRESSED
            decomp = zlib.decompress(data[offset+8:offset+8+size])
            idx = decomp.find(b"Pose_Para")
            if idx != -1:
                # Cari tag tipe data array di sekitar Pose_Para
                for i in range(idx, min(idx + 120, len(decomp) - 36)):
                    t, sz = struct.unpack("<II", decomp[i:i+8])
                    # t=7 adalah miSINGLE (float32)
                    if t == 7 and sz >= 28:
                        vals = struct.unpack("<7f", decomp[i+8:i+8+28])
                        return math.degrees(vals[0]), math.degrees(vals[1]), math.degrees(vals[2])
                    # t=9 adalah miDOUBLE (float64)
                    elif t == 9 and sz >= 56:
                        vals = struct.unpack("<7d", decomp[i+8:i+8+56])
                        return math.degrees(vals[0]), math.degrees(vals[1]), math.degrees(vals[2])
        offset += 8 + size
    raise ValueError(f"Tidak dapat membaca Pose_Para dari biner .mat: {mat_path}")

def read_pose_mat(mat_path: Path or str) -> Tuple[float, float, float]:
    """
    Membaca file pose .mat dari 300W-LP atau AFLW2000-3D.
    Mengembalikan (pitch_deg, yaw_deg, roll_deg) dalam satuan derajat (°).
    Mendukung format 'Pose_Para' (vektor 3DDFA), 'Pose_Pitch', maupun variasi huruf kecil.
    """
    mat_path = Path(mat_path)
    if HAS_SCIPY:
        try:
            m = sio.loadmat(str(mat_path))
            if "Pose_Para" in m:
                arr = np.asarray(m["Pose_Para"]).squeeze()
                pitch = float(np.degrees(arr[0]))
                yaw   = float(np.degrees(arr[1]))
                roll  = float(np.degrees(arr[2]))
                return pitch, yaw, roll

            def get_val(*keys):
                for k in keys:
                    if k in m:
                        return float(np.asarray(m[k]).squeeze())
                return None

            p = get_val("Pose_Pitch", "pitch", "PosePitch")
            y = get_val("Pose_Yaw", "yaw", "PoseYaw")
            r = get_val("Pose_Roll", "roll", "PoseRoll")

            if p is not None and y is not None and r is not None:
                return float(np.degrees(p)), float(np.degrees(y)), float(np.degrees(r))
        except Exception:
            pass

    return _parse_mat_pure_python(mat_path)

# =============================================================================
# 3. GROUP IDENTIFIER & ANTI-DATA LEAKAGE ENGINE (T1: 300W-LP)
# =============================================================================

def group_id_from_name(fname: str) -> str:
    """
    Mengekstrak ID group dari nama file dengan membuang suffix rotasi/flip.
    Contoh:
      - AFW_1051618982_1_0.jpg       -> AFW_1051618982_1
      - HELEN_100032540_1_15.jpg     -> HELEN_100032540_1
      - LFPW_image_train_0001_0.jpg  -> LFPW_image_train_0001
      - LFPW_image_test_0223_7.jpg   -> LFPW_image_test_0223
    Dengan memotong elemen terakhir setelah '_', seluruh variasi rotasi dan mirror
    dari 1 foto asli yang sama dipastikan memiliki ID group yang identik.
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem

def build_lp_index(include_flips: bool = False) -> List[Dict[str, str]]:
    """
    Memindai dataset 300W-LP secara cepat menggunakan os.scandir.
    Hanya mengindeks file jpg yang memiliki pasangan .mat.
    """
    lp_root = get_300wlp_root()
    ds_names = ["AFW", "HELEN", "IBUG", "LFPW"]
    if include_flips:
        ds_names += ["AFW_Flip", "HELEN_Flip", "IBUG_Flip", "LFPW_Flip"]

    rows = []
    for ds in ds_names:
        d = lp_root / ds
        if not d.exists():
            continue
        
        # Kumpulkan nama file dalam set untuk lookup O(1) cepat
        files = set()
        with os.scandir(str(d)) as it:
            for entry in it:
                if entry.is_file():
                    files.add(entry.name)

        # Pasangkan .jpg dengan .mat
        d_str = str(d)
        for fname in sorted(files):
            if fname.lower().endswith(".jpg"):
                mat_name = fname[:-4] + ".mat"
                if mat_name in files:
                    jpg_path = os.path.join(d_str, fname)
                    mat_path = os.path.join(d_str, mat_name)
                    rows.append({
                        "jpg": jpg_path,
                        "mat": mat_path,
                        "group": group_id_from_name(fname),
                        "dataset_source": ds
                    })
    return rows

def build_test_index() -> List[Dict[str, str]]:
    """
    Memindai dataset AFLW2000-3D yang sakral (hanya dipakai sekali di evaluasi akhir).
    """
    aflw_root = get_aflw2000_root()
    root_str = str(aflw_root)
    files = set()
    with os.scandir(root_str) as it:
        for entry in it:
            if entry.is_file():
                files.add(entry.name)

    rows = []
    for fname in sorted(files):
        if fname.lower().endswith(".jpg"):
            mat_name = fname[:-4] + ".mat"
            if mat_name in files:
                jpg_path = os.path.join(root_str, fname)
                mat_path = os.path.join(root_str, mat_name)
                rows.append({
                    "jpg": jpg_path,
                    "mat": mat_path,
                    "group": Path(fname).stem,
                    "dataset_source": "AFLW2000"
                })
    return rows

def group_split(
    rows: List[Dict[str, str]], 
    val_frac: float = 0.10, 
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Membagi dataset BERBASIS GROUP secara ketat untuk mencegah data leakage.
    Seluruh gambar dari subjek yang sama hanya boleh berada di Train ATAU Val, tidak boleh keduanya.
    Irisan group wajib KOSONG (0).
    """
    unique_groups = sorted({r["group"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_val = int(len(unique_groups) * val_frac)
    val_groups = set(unique_groups[:n_val])
    train_groups = set(unique_groups[n_val:])

    # VERIFIKASI WAJIB: Irisan group harus 0!
    leakage = train_groups & val_groups
    assert len(leakage) == 0, f"KRITIS: Terjadi DATA LEAKAGE! Irisan group: {leakage}"

    train_rows = [r for r in rows if r["group"] in train_groups]
    val_rows = [r for r in rows if r["group"] in val_groups]

    # Double-check level sampel
    check_leak = {r["group"] for r in train_rows} & {r["group"] for r in val_rows}
    assert len(check_leak) == 0, "KRITIS: Terjadi data leakage pada sampel split!"

    return train_rows, val_rows

# =============================================================================
# 4. STRATIFIED SPLIT ENGINE (T2: MYOPIA FUNDUS)
# =============================================================================

def build_myopia_index() -> List[Dict[str, Any]]:
    """
    Memindai direktori Myopia images dengan kecepatan tinggi (os.scandir):
    - Myopia_images/ -> kelas 1 (label: 1.0)
    - Normal_images/ -> kelas 0 (label: 0.0)
    """
    root = get_myopia_root()
    myopia_dir = root / "Myopia_images"
    normal_dir = root / "Normal_images"

    if not myopia_dir.exists() or not normal_dir.exists():
        raise FileNotFoundError(
            f"Folder Myopia_images atau Normal_images tidak ditemukan di {root}"
        )

    rows = []
    # Pindai Myopia images (kelas 1)
    with os.scandir(str(myopia_dir)) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({
                    "img": entry.path,
                    "label": 1,
                    "class_name": "myopia",
                    "filename": entry.name
                })

    # Pindai Normal images (kelas 0)
    with os.scandir(str(normal_dir)) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({
                    "img": entry.path,
                    "label": 0,
                    "class_name": "normal",
                    "filename": entry.name
                })
    return rows

def stratified_myopia_split(
    rows: List[Dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Melakukan Stratified Split 70/15/15 secara seimbang per kelas.
    Menjamin tidak ada file yang terpecah split antar train/val/test.
    """
    assert math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-5), "Total fraksi split harus 1.0"

    myopia_rows = [r for r in rows if r["label"] == 1]
    normal_rows = [r for r in rows if r["label"] == 0]

    # Urutkan terlebih dahulu untuk konsistensi deterministik lintas platform
    myopia_rows.sort(key=lambda r: r["filename"])
    normal_rows.sort(key=lambda r: r["filename"])

    rng = random.Random(seed)
    rng.shuffle(myopia_rows)
    rng.shuffle(normal_rows)

    def split_list(lst):
        n_tr = int(len(lst) * train_frac)
        n_va = int(len(lst) * val_frac)
        tr = lst[:n_tr]
        va = lst[n_tr:n_tr + n_va]
        te = lst[n_tr + n_va:]
        return tr, va, te

    m_tr, m_va, m_te = split_list(myopia_rows)
    n_tr, n_va, n_te = split_list(normal_rows)

    train_rows = m_tr + n_tr
    val_rows = m_va + n_va
    test_rows = m_te + n_te

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    # Verifikasi integritas: tidak boleh ada file yang tumpang tindih
    tr_set = {r["img"] for r in train_rows}
    va_set = {r["img"] for r in val_rows}
    te_set = {r["img"] for r in test_rows}

    assert not (tr_set & va_set), "LEAKAGE: Ada file sama di Train dan Val!"
    assert not (tr_set & te_set), "LEAKAGE: Ada file sama di Train dan Test!"
    assert not (va_set & te_set), "LEAKAGE: Ada file sama di Val dan Test!"

    return train_rows, val_rows, test_rows

# =============================================================================
# 5. HARDWARE DIAGNOSTICS & OPTIMIZATION REPORT
# =============================================================================

def print_hardware_summary():
    """
    Mencetak ringkasan kapabilitas hardware CPU Intel Core i7-13620H & GPU NVIDIA RTX 5050.
    """
    print("\n" + "=" * 70)
    print(" 🚀 JAGAMATA HARDWARE & ENVIRONMENT PROFILE")
    print("=" * 70)
    print(f" Platform OS        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" Processor (CPU)    : {platform.processor() or 'Intel Core i7-13620H'}")
    print(f" Logical CPU Cores  : {os.cpu_count()} threads (optimal num_workers=8)")
    
    try:
        import torch
        print(f" PyTorch Version    : {torch.__version__}")
        print(f" CUDA Available     : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            cap = torch.cuda.get_device_capability(0)
            arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
            sm_str = f"sm_{cap[0]}{cap[1]}"
            print(f" GPU Device         : {gpu_name} ({vram_mb:.0f} MB VRAM)")
            print(f" CUDA Device Cap.   : {cap} ({sm_str})")
            print(f" cuDNN Enabled      : {torch.backends.cudnn.enabled}")
            print(f" cuDNN Benchmark    : True (Akselerasi konvolusi diaktifkan)")
            print(f" AMP Support        : Tersedia (Memanfaatkan RTX Tensor Cores)")

            if arch_list and sm_str not in arch_list and cap[0] >= 10:
                print(f"\n [⚠️ CATATAN ARSITEKTUR GPU]:")
                print(f"  GPU Anda ({gpu_name}) berbasis arsitektur NVIDIA Blackwell ({sm_str}).")
                print(f"  Versi PyTorch saat ini belum memuat kernel native {sm_str}.")
                print(f"  Solusi: Perbarui PyTorch ke build CUDA 12.8 (cu128) dengan perintah:")
                print(f"    pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        else:
            print(" [PERINGATAN] CUDA tidak terdeteksi! Training akan fallback ke CPU.")
    except ImportError:
        print(" PyTorch Version    : [Belum terinstall di lingkungan ini]")
    print("=" * 70 + "\n")

# =============================================================================
# 6. EVALUATION METRICS ENGINE
# =============================================================================

def compute_regression_metrics(preds: Any, tgts: Any) -> Dict[str, float]:
    """Menghitung metrik regresi (MAE, RMSE, Median AE, Max Error)."""
    global np
    if np is None:
        import numpy as np
    preds_arr = np.asarray(preds, dtype=np.float64)
    tgts_arr = np.asarray(tgts, dtype=np.float64)
    errors = np.abs(preds_arr - tgts_arr)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((preds_arr - tgts_arr) ** 2)))
    medae = float(np.median(errors))
    max_err = float(np.max(errors))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "median_ae": round(medae, 4),
        "max_error": round(max_err, 4)
    }

def bootstrap_mae_ci95(
    preds: Any, 
    tgts: Any, 
    n_resamples: int = 1000, 
    seed: int = 42
) -> Tuple[float, float]:
    """
    Menghitung 95% Confidence Interval untuk MAE menggunakan Bootstrap (1000 resamples).
    Sesuai standar evaluasi sakral AFLW2000-3D di Panduan §5 baris 351.
    """
    global np
    if np is None:
        import numpy as np
    preds_arr = np.asarray(preds, dtype=np.float64).ravel()
    tgts_arr = np.asarray(tgts, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n = len(preds_arr)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_maes = np.mean(np.abs(preds_arr[indices] - tgts_arr[indices]), axis=1)
    ci_low, ci_high = np.percentile(boot_maes, [2.5, 97.5])
    return float(ci_low), float(ci_high)

def compute_classification_metrics(
    y_true: Any, 
    y_prob: Any, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Menghitung metrik klasifikasi klinis: AUC, F1-Score, Sensitivity (Recall), Specificity, Accuracy.
    """
    global np
    if np is None:
        import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_p >= threshold).astype(int)

    auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    acc = float(accuracy_score(y_t, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(acc, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

def bootstrap_auc_ci95(
    y_true: Any, 
    y_prob: Any, 
    n_resamples: int = 1000, 
    seed: int = 42
) -> Tuple[float, float]:
    """
    Menghitung 95% Confidence Interval untuk ROC-AUC menggunakan Bootstrap (1000 resamples).
    """
    global np
    if np is None:
        import numpy as np
    from sklearn.metrics import roc_auc_score
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n = len(y_t)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_true = y_t[idx]
        sample_prob = y_p[idx]
        if len(np.unique(sample_true)) > 1:
            aucs.append(roc_auc_score(sample_true, sample_prob))
    if not aucs:
        return 0.0, 0.0
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return float(ci_low), float(ci_high)

# =============================================================================
# CLI TEST & GENERATE SPLIT JSON
# =============================================================================

def main():
    print_hardware_summary()

    print("1. Memeriksa dataset 300W-LP (T1)...")
    try:
        lp_root = get_300wlp_root()
        print(f"   Root 300W-LP terdeteksi: {lp_root}")
        t0 = time.time()
        lp_rows = build_lp_index(include_flips=False)
        lp_groups = {r["group"] for r in lp_rows}
        print(f"   Ditemukan {len(lp_rows)} pasang gambar asli + .mat ({len(lp_groups)} group unik) [{time.time() - t0:.2f}s]")

        train_lp, val_lp = group_split(lp_rows, val_frac=0.10, seed=42)
        print(f"   Train: {len(train_lp)} sampel, Val: {len(val_lp)} sampel")
        print("   Verifikasi irisan group:", len({r['group'] for r in train_lp} & {r['group'] for r in val_lp}))

        test_aflw = build_test_index()
        print(f"   AFLW2000-3D (Test sakral): {len(test_aflw)} sampel")

        split_pitch_file = Path("ml/train/split_pitch.json")
        split_pitch_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_pitch_file, "w") as f:
            json.dump({"train": train_lp, "val": val_lp, "test": test_aflw}, f)
        print(f"   -> {split_pitch_file} berhasil disimpan ({len(train_lp)} train, {len(val_lp)} val, {len(test_aflw)} test)!")
    except Exception as e:
        print(f"   [SKIP 300W-LP]: {e}")

    print("\n2. Memeriksa dataset Myopia Images (T2)...")
    try:
        myopia_root = get_myopia_root()
        print(f"   Root Myopia Images terdeteksi: {myopia_root}")
        t0 = time.time()
        myopia_rows = build_myopia_index()
        print(f"   Total gambar fundus: {len(myopia_rows)} [{time.time() - t0:.2f}s]")

        t1 = time.time()
        m_tr, m_va, m_te = stratified_myopia_split(myopia_rows, seed=42)
        print(f"   Train: {len(m_tr)}, Val: {len(m_va)}, Test: {len(m_te)} [{time.time() - t1:.2f}s]")

        split_fundus_file = Path("ml/train/split_fundus.json")
        split_fundus_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_fundus_file, "w") as f:
            json.dump({"train": m_tr, "val": m_va, "test": m_te}, f)
        print(f"   -> {split_fundus_file} berhasil disimpan ({len(m_tr)} train, {len(m_va)} val, {len(m_te)} test)!")
    except Exception as e:
        print(f"   [SKIP Myopia]: {e}")

if __name__ == "__main__":
    main()
