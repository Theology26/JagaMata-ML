"""
=============================================================================
JagaMata ML Training Suite - Master Orchestrator (T1 & T2 Sequential Run)
=============================================================================
Menjalankan training otomatis berurutan (cocok untuk run semalaman / overnight):
1. Task 1 (T1): Head Pitch Regressor (MobileNetV3-Small)
   - 300W-LP train/val (Group split, irisan=0)
   - Evaluasi AFLW2000-3D (test sakral sekali pakai) -> MAE Pitch <= 5.0°
2. Task 2 (T2): Myopia Fundus Classifier (EfficientNet-B0)
   - Myopia Images 70/15/15 stratified split
   - Evaluasi Test set -> ROC-AUC >= 0.90
3. Auto-generate ml/LAPORAN_TRAINING.md dengan angka hasil nyata
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path

# Sisipkan path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pitch import train_t1_pitch, parse_args as parse_args_pitch
from train_fundus import train_t2_fundus, parse_args as parse_args_fundus

def run_all():
    print("\n" + "=" * 80)
    print(" 🚀 MEMULAI PIPELINE TRAINING LENGKAP JAGAMATA (TASK 1 & TASK 2)")
    print(" Hardware: NVIDIA GeForce RTX 5050 (AMP FP16) + Intel Core i7-13620H (16 Threads)")
    print("=" * 80 + "\n")

    t_start = time.time()

    # ---------------------------------------------------------
    # 1. TASK 1: HEAD PITCH REGRESSOR
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> FASE 1: TRAINING TASK 1 - HEAD PITCH REGRESSOR (MobileNetV3-Small)")
    print("#" * 80 + "\n")

    sys.argv = [
        "train_pitch.py",
        "--batch-size", "64",
        "--epochs", "30",
        "--num-workers", "8",
        "--amp",
    ]
    args_t1 = parse_args_pitch()
    train_t1_pitch(args_t1)

    # ---------------------------------------------------------
    # 2. TASK 2: MYOPIA FUNDUS CLASSIFIER
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print(" >>> FASE 2: TRAINING TASK 2 - MYOPIA FUNDUS CLASSIFIER (EfficientNet-B0)")
    print("#" * 80 + "\n")

    sys.argv = [
        "train_fundus.py",
        "--batch-size", "64",
        "--epochs", "20",
        "--num-workers", "8",
        "--amp",
    ]
    args_t2 = parse_args_fundus()
    train_t2_fundus(args_t2)

    total_duration = time.time() - t_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print("\n" + "=" * 80)
    print(f" 🏁 SELURUH TRAINING BERHASIL SELESAI DALAM {hours}h {minutes}m {seconds}s! 🎉")
    print(" Artefak tersimpan rapi di direktori ml/runs/")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_all()
