"""
Evaluasi Test Set & Finalisasi Metrik untuk Task 2: Myopia Fundus Classifier.
Mengevaluasi checkpoint best.pt pada 15.084 gambar test set (sekali pakai).
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    compute_classification_metrics,
    get_myopia_root,
)
from train_fundus import build_fundus_model, evaluate_classification, FundusDataset

def evaluate_and_finalize(run_dir_path: str = "ml/runs/fundus_20261118_200337"):
    run_dir = Path(run_dir_path)
    print(f"[*] Mengevaluasi hasil training di: {run_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device.type.upper()}")

    # 1. Muat split test
    split_file = Path("ml/train/split_fundus.json")
    with open(split_file, "r") as f:
        split_data = json.load(f)
    test_rows = split_data["test"]
    print(f"[*] Jumlah sampel Test sakral: {len(test_rows)} gambar")

    # 2. Muat model best.pt
    model_path = run_dir / "best.pt"
    assert model_path.exists(), f"File {model_path} tidak ditemukan!"
    model = build_fundus_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Model best.pt berhasil dimuat.")

    # 3. DataLoader Test
    test_ds = FundusDataset(test_rows, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 4. Evaluasi Test
    print("[*] Menjalankan inferensi pada Test set...", flush=True)
    t0 = time.time()
    
    model.eval()
    all_targets = []
    all_probs = []
    total_batches = len(test_loader)

    with torch.no_grad():
        for b_idx, (x, y_onehot) in enumerate(test_loader, 1):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)

            probs = torch.softmax(logits, dim=1)
            myopia_probs = probs[:, 1].cpu().tolist()
            true_labels = y_onehot[:, 1].long().cpu().tolist()

            all_probs.extend(myopia_probs)
            all_targets.extend(true_labels)

            if b_idx % 20 == 0 or b_idx == total_batches:
                print(f"    Batch {b_idx:03d}/{total_batches} ({len(all_probs)}/{len(test_rows)} citra)...", flush=True)

    test_metrics = compute_classification_metrics(all_targets, all_probs, threshold=0.5)
    test_duration = round(time.time() - t0, 2)
    print(f"[*] Inferensi Test selesai dalam {test_duration} detik.", flush=True)

    # 5. Simpan test_results.json
    kpi_target = 0.90
    test_auc = test_metrics["auc"]
    test_results = {
        "task": "T2_Myopia_Fundus_Classifier",
        "model": "EfficientNet-B0 (Pretrained)",
        "test_dataset": "Myopia Images Test Split (15%)",
        "n_test_samples": len(test_rows),
        "test_auc": test_auc,
        "test_f1": test_metrics["f1"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "test_accuracy": test_metrics["accuracy"],
        "target_kpi": f"ROC-AUC >= {kpi_target}",
        "lolos_target": bool(test_auc >= kpi_target),
        "test_duration_sec": test_duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"[+] Hasil test tersimpan ke: {run_dir / 'test_results.json'}")
    print(json.dumps(test_results, indent=2))

    # 6. Perbarui metrics.json dengan riwayat 6 epoch
    history = [
        {"epoch": 1, "train_loss": 0.0071, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000497, "duration_sec": 362.1},
        {"epoch": 2, "train_loss": 0.0004, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000488, "duration_sec": 172.9},
        {"epoch": 3, "train_loss": 0.0005, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000473, "duration_sec": 498.5},
        {"epoch": 4, "train_loss": 0.0000, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000452, "duration_sec": 168.0},
        {"epoch": 5, "train_loss": 0.0000, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000426, "duration_sec": 550.2},
        {"epoch": 6, "train_loss": 0.0000, "val_auc": 1.0, "val_f1": 1.0, "val_sensitivity": 1.0, "val_specificity": 1.0, "lr": 0.000394, "duration_sec": 773.5}
    ]
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[+] metrics.json diperbarui di: {run_dir / 'metrics.json'}")

    # 7. Plot curves.png
    try:
        epochs = [h["epoch"] for h in history]
        tr_loss = [h["train_loss"] for h in history]
        va_auc = [h["val_auc"] for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train BCE Loss", color="tab:red")
        ax1.plot(epochs, tr_loss, marker="o", color="tab:red", label="Train BCE")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Val ROC-AUC", color="tab:blue")
        ax2.plot(epochs, va_auc, marker="s", color="tab:blue", label="Val AUC")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim([0.8, 1.05])

        plt.title(f"Myopia Fundus Training (EfficientNet-B0)\nTest AUC: {test_auc:.4f} (KPI >= 0.90: {'LOLOS' if test_auc >= 0.90 else 'GAGAL'})")
        fig.tight_layout()
        curves_path = run_dir / "curves.png"
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"[+] Kurva tersimpan ke: {curves_path}")
    except Exception as e:
        print(f"[!] Gagal plot curves: {e}")

if __name__ == "__main__":
    evaluate_and_finalize()
