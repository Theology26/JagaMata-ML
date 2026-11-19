"""
=============================================================================
JagaMata ML Training Suite - Hyperparameter Tuning & Ablation Engine
=============================================================================
Eksplorasi sistematis hyperparameter model Head Pitch Regressor sesuai 
PANDUAN_TRAINING_ML_JagaMata.md §6:
1. Satu run = satu hipotesis terarah.
2. Seleksi model terbaik berbasis VALIDATION MAE (bukan test set).
3. Hanya model pemenang validasi yang dievaluasi pada AFLW2000-3D test set.
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pitch_tuned import train_t1_pitch_tuned, parse_args

EXPERIMENTS = [
    {
        "name": "pitch_exp_warmup_lr5e4",
        "description": "Hipotesis Warmup: LR 5e-4 (konservatif) + 3 Epoch Linear Warmup + Beta 1.2",
        "args": [
            "--run-name", "pitch_exp_warmup_lr5e4",
            "--lr", "5e-4",
            "--warmup-epochs", "3",
            "--weight-decay", "1e-3",
            "--dropout", "0.3",
            "--beta", "1.2",
            "--pitch-weight", "1.5",
            "--epochs", "10",
            "--patience", "5",
            "--batch-size", "64",
            "--num-workers", "8",
            "--amp",
            "--skip-sanity"
        ]
    },
    {
        "name": "pitch_exp_pitchweight2_beta1",
        "description": "Hipotesis Presisi: Pitch Weight 2.0x (fokus pitch) + Huber Beta 1.0 (penalti ketat)",
        "args": [
            "--run-name", "pitch_exp_pitchweight2_beta1",
            "--lr", "1e-3",
            "--warmup-epochs", "0",
            "--weight-decay", "1e-3",
            "--dropout", "0.3",
            "--beta", "1.0",
            "--pitch-weight", "2.0",
            "--epochs", "10",
            "--patience", "5",
            "--batch-size", "64",
            "--num-workers", "8",
            "--amp",
            "--skip-sanity"
        ]
    }
]

def run_tuning():
    print("\n" + "=" * 80)
    print(" 🚀 MEMULAI SYSTEMATIC HYPERPARAMETER SEARCH - JAGAMATA T1 PITCH")
    print(" Standar Protokol: Panduan §6 (Ablation Run, Seleksi via Validation MAE)")
    print("=" * 80 + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_results": [
            {
                "run_name": "pitch_20261118_181059",
                "label": "Baseline Single-Head (lr 1e-3, wd 1e-4, beta 1.0)",
                "val_pitch_mae_best": 2.4255,
                "val_screen_mae_best": 2.45,
                "notes": "Arsitektur awal, mengalami oversmoothing pada yaw ekstrem"
            },
            {
                "run_name": "pitch_tuned_20261118_210456",
                "label": "Tuned Multi-Task v1 (lr 1e-3, wd 1e-3, dropout 0.3, beta 1.5, pitch_wt 1.5)",
                "val_pitch_mae_best": 2.1498,
                "val_screen_mae_best": 2.50,
                "notes": "Multi-task [pitch, yaw, roll], pangkas error yaw ekstrem 52.6%"
            }
        ]
    }

    for exp in EXPERIMENTS:
        print("\n" + "#" * 80)
        print(f" >>> MENJALANKAN EKSPERIMEN: {exp['name']}")
        print(f" Deskripsi: {exp['description']}")
        print("#" * 80 + "\n")

        # Set argumen CLI
        sys.argv = ["train_pitch_tuned.py"] + exp["args"]
        parsed = parse_args()
        
        t0 = time.time()
        train_t1_pitch_tuned(parsed)
        dur_min = round((time.time() - t0) / 60, 2)

        # Baca hasil dari run folder
        run_folder = Path("ml/runs") / exp["name"]
        metrics_file = run_folder / "metrics.json"
        best_val = None
        best_screen = None
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                history = json.load(f)
            if history:
                best_val = min(h["val_pitch_mae"] for h in history)
                best_screen = min(h["val_pitch_screen_mae"] for h in history)

        record = {
            "run_name": exp["name"],
            "label": exp["description"],
            "val_pitch_mae_best": best_val,
            "val_screen_mae_best": best_screen,
            "duration_minutes": dur_min,
            "notes": f"Tuned run selesai dalam {dur_min} menit"
        }
        summary["benchmark_results"].append(record)

        # Simpan ringkasan progres
        summary_file = Path("ml/runs/hyperparameter_tuning_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[+] Ringkasan benchmarking diperbarui -> {summary_file}")

    print("\n" + "=" * 80)
    print(" 🏁 HYPERPARAMETER SEARCH SELESAI!")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_tuning()
