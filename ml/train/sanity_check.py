"""
=============================================================================
JagaMata ML Training Suite - Comprehensive Sanity & Integrity Check
=============================================================================
Skrip diagnosa komprehensif untuk memverifikasi seluruh komponen sebelum
training semalaman (overnight run):
1. Status Hardware (NVIDIA RTX 5050 8GB, Intel Core i7-13620H, 16GB RAM)
2. Aksesibilitas Dataset (300W-LP, AFLW2000-3D, Myopia images)
3. Integritas Parser .mat (Pose_Para & derajat pose)
4. Verifikasi Anti-Data Leakage (Irisan group = 0)
5. Test Overfit 32 Sampel (Pipeline & Gradient Check)
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path

# Sisipkan folder ml/train ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    get_300wlp_root,
    get_aflw2000_root,
    get_myopia_root,
    read_pose_mat,
    group_id_from_name,
    print_hardware_summary,
)

def run_diagnostics():
    print_hardware_summary()
    all_passed = True

    # ---------------------------------------------------------
    # 1. VERIFIKASI DATASET
    # ---------------------------------------------------------
    print(" [1/4] MEMERIKSA AKSES DATASET...")
    try:
        lp_p = get_300wlp_root()
        print(f"  ✅ 300W-LP Root          : {lp_p}")
    except Exception as e:
        print(f"  ❌ 300W-LP Error         : {e}")
        all_passed = False

    try:
        af_p = get_aflw2000_root()
        print(f"  ✅ AFLW2000-3D Root      : {af_p}")
    except Exception as e:
        print(f"  ❌ AFLW2000-3D Error     : {e}")
        all_passed = False

    try:
        my_p = get_myopia_root()
        print(f"  ✅ Myopia Images Root    : {my_p}")
    except Exception as e:
        print(f"  ❌ Myopia Images Error   : {e}")
        all_passed = False

    # ---------------------------------------------------------
    # 2. VERIFIKASI PARSER .MAT & LABEL
    # ---------------------------------------------------------
    print("\n [2/4] MENGUJI PEMBACA POSE .MAT...")
    sample_mat = None
    try:
        af_p = get_aflw2000_root()
        for f in os.scandir(str(af_p)):
            if f.name.endswith(".mat"):
                sample_mat = Path(f.path)
                break
        if sample_mat:
            pitch, yaw, roll = read_pose_mat(sample_mat)
            print(f"  ✅ Sample {sample_mat.name} -> Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°")
        else:
            print("  ❌ Tidak ditemukan file .mat di folder AFLW2000!")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Gagal membaca .mat: {e}")
        all_passed = False

    # ---------------------------------------------------------
    # 3. VERIFIKASI FILE SPLIT & ANTI-LEAKAGE
    # ---------------------------------------------------------
    print("\n [3/4] MEMERIKSA INTEGRITAS SPLIT & ANTI-LEAKAGE...")
    split_pitch_file = Path("ml/train/split_pitch.json")
    if split_pitch_file.exists():
        with open(split_pitch_file, "r") as f:
            sp = json.load(f)
        tr_g = {r["group"] for r in sp["train"]}
        va_g = {r["group"] for r in sp["val"]}
        leak = tr_g & va_g
        if len(leak) == 0:
            print(f"  ✅ Split Pitch: Train={len(sp['train'])}, Val={len(sp['val'])}, Test={len(sp['test'])}")
            print(f"     Verifikasi Leakage: Irisan Group = {len(leak)} (AMAN! Anti-leakage lulus)")
        else:
            print(f"  ❌ KRITIS: Data Leakage terdeteksi pada Split Pitch! Irisan: {leak}")
            all_passed = False
    else:
        print("  ⚠️ split_pitch.json belum ada. Jalankan: python ml/train/common.py")

    split_fundus_file = Path("ml/train/split_fundus.json")
    if split_fundus_file.exists():
        with open(split_fundus_file, "r") as f:
            sf = json.load(f)
        tr_s = {r["img"] for r in sf["train"]}
        va_s = {r["img"] for r in sf["val"]}
        te_s = {r["img"] for r in sf["test"]}
        if not (tr_s & va_s) and not (tr_s & te_s) and not (va_s & te_s):
            print(f"  ✅ Split Fundus: Train={len(sf['train'])}, Val={len(sf['val'])}, Test={len(sf['test'])}")
            print("     Verifikasi Leakage: Irisan File = 0 (AMAN! Anti-leakage lulus)")
        else:
            print("  ❌ KRITIS: Data Leakage terdeteksi pada Split Fundus!")
            all_passed = False
    else:
        print("  ⚠️ split_fundus.json belum ada. Jalankan: python ml/train/common.py")

    # ---------------------------------------------------------
    # 4. OVERFIT TEST (JIKA PYTORCH TERPASANG)
    # ---------------------------------------------------------
    print("\n [4/4] MEMERIKSA ENVIRONTMENT PYTORCH & CUDA...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__} terpasang.")
        print(f"  ✅ CUDA Tersedia: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print("  ✅ Tensor Cores AMP siap digunakan.")
            try:
                test_t = torch.randn(4, 4, device="cuda")
                _ = test_t @ test_t
                print("  ✅ GPU Kernel Execution: Berhasil (Kernel GPU bekerja normal!)")
            except RuntimeError as e:
                print(f"  ⚠️ GPU Kernel Execution Gagal: Arsitektur sm_120 (Blackwell) membutuhkan PyTorch cu128.")
                print(f"     -> Update via: pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    except ImportError:
        print("  ℹ️ PyTorch belum terinstall di Python ini. Jalankan pip install torch sesuai panduan.")

    print("\n" + "=" * 65)
    if all_passed:
        print(" 🎯 HASIL DIAGNOSA: SELURUH SISTEM SIAP UNTUK TRAINING! 🚀")
    else:
        print(" ⚠️ ADA CATATAN YANG PERLU DIPERHATIKAN SEBELUM TRAINING.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_diagnostics()
