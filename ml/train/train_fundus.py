"""
=============================================================================
JagaMata ML Training Suite - Task 2 (T2): Myopia Fundus Classifier
=============================================================================
Arsitektur    : EfficientNet-B0 (Pretrained ImageNet) -> 2 Output Logits (Myopia vs Normal)
Dataset       : Myopia Images (kellysanderson/myopia-images)
                - Kelas 1: Images/Myopia_images/
                - Kelas 0: Images/Normal_images/
Aturan Split  : Stratified Split 70% Train / 15% Val / 15% Test (per kelas, anti-leakage)
Loss Function : BCEWithLogitsLoss (Multi-label / One-hot target formulation)
Optimizer     : AdamW (lr=5e-4, weight_decay=1e-4)
Scheduler     : CosineAnnealingLR
Hardware Opt  : CUDA, AMP (FP16 Tensor Cores RTX 5050), num_workers=8, pin_memory=True
Metrik Target : ROC-AUC >= 0.90 (Evaluasi Bootstrap 95% CI 1000 resamples)
                + F1-Score, Sensitivity (Recall), Specificity, Accuracy
=============================================================================
"""

import os
import sys
import time
import math
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Import modul helper common dari package lokal
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    build_myopia_index,
    stratified_myopia_split,
    compute_classification_metrics,
    bootstrap_auc_ci95,
    print_hardware_summary,
)

# Deterministic Seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# DATASET & AUGMENTASI FUNDUS
# =============================================================================

class FundusDataset(Dataset):
    """
    Dataset loader untuk citra fundus retina (Normal vs Myopia).
    Augmentasi dirancang aman untuk citra retina sirkular (rotasi, flip, affine halus).
    """
    def __init__(self, rows: List[Dict[str, Any]], augment: bool = False):
        self.rows = rows
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        img_path = row["img"]
        label = row["label"]  # 1: Myopia, 0: Normal

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor_img = self.transform(img)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)

        # One-hot target untuk BCEWithLogitsLoss (num_classes=2):
        # Kelas 0: [1.0, 0.0], Kelas 1: [0.0, 1.0]
        one_hot = torch.tensor([1.0 - label, float(label)], dtype=torch.float32)

        return tensor_img, one_hot

# =============================================================================
# ARSITEKTUR MODEL (EfficientNet-B0 -> num_classes=2)
# =============================================================================

def build_fundus_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Membangun model klasifikasi EfficientNet-B0.
    Mendukung timm jika terpasang, atau torchvision sebagai fallback.
    """
    try:
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception:
        pass

    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# =============================================================================
# SANITY CHECK / OVERFIT TEST (32 SAMPEL)
# =============================================================================

def run_fundus_sanity_check(
    train_rows: List[Dict[str, Any]], 
    device: torch.device, 
    use_amp: bool = True
) -> bool:
    """
    Sanity check wajib pada 32 sampel fundus selama 25 epoch.
    Memastikan pipeline data, gradient, dan loss BCEWithLogitsLoss turun >85%.
    """
    print("\n" + "=" * 65)
    print(" 🧪 MENJALANKAN SANITY CHECK FUNDUS (OVERFIT TEST: 32 SAMPEL)")
    print("=" * 65)

    sample_rows = train_rows[:32]
    dataset = FundusDataset(sample_rows, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_fundus_model(num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    initial_loss = None
    final_loss = None

    model.train()
    try:
        for ep in range(1, 26):
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = loss_fn(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * len(x)

            avg_loss = total_loss / len(sample_rows)
            if ep == 1:
                initial_loss = avg_loss
            final_loss = avg_loss

            if ep % 5 == 0 or ep == 25:
                print(f"   [Sanity Epoch {ep:02d}/25] Loss: {avg_loss:.4f}", flush=True)
    except RuntimeError as e:
        if "no kernel image is available" in str(e).lower() or "sm_120" in str(e).lower():
            print("\n" + "!" * 75)
            print(" [ERROR KOMPATIBILITAS ARSITEKTUR GPU RTX 5050 (sm_120)]")
            print("!" * 75)
            print(" Penyebab : NVIDIA GeForce RTX 5050 menggunakan arsitektur Blackwell (sm_120),")
            print("            sedangkan build PyTorch cu124 hanya memuat kernel hingga sm_90.")
            print("\n Solusi   : Perbarui PyTorch ke build CUDA 12.8 (cu128) dengan perintah:")
            print("   pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
            print("\n Alternatif sementara (menjalankan via CPU Intel Core i7 16-thread):")
            print("   python ml\\train\\train_fundus.py --device cpu")
            print("!" * 75 + "\n")
            return False
        raise e

    ratio = final_loss / (initial_loss + 1e-7)
    print(f"   Loss Awal: {initial_loss:.4f} -> Loss Akhir: {final_loss:.4f} (Rasio: {ratio:.1%})")

    passed = ratio < 0.15
    if passed:
        print("   ✅ SANITY CHECK FUNDUS LULUS! Pipeline normal & gradient berjalan lancar.")
    else:
        print("   ❌ SANITY CHECK GAGAL! Periksa normalisasi atau learning rate.")
    print("=" * 65 + "\n")
    return passed

# =============================================================================
# EVALUASI VALIDASI / TEST
# =============================================================================

def evaluate_classification(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    use_amp: bool = True
) -> Tuple[Dict[str, float], List[int], List[float]]:
    """
    Evaluasi model untuk mendapatkan AUC, F1, Sensitivity, Specificity, dan Accuracy.
    Probabilitas kelas positif dihitung dengan Softmax pada logit kedua (indeks 1).
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y_onehot in loader:
            x = x.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)

            # Softmax probabilitas
            probs = torch.softmax(logits, dim=1)
            # Indeks 1 adalah probabilitas Myopia (kelas positif)
            myopia_probs = probs[:, 1].cpu().tolist()
            # Indeks 1 pada one-hot adalah target biner sesungguhnya
            true_labels = y_onehot[:, 1].long().cpu().tolist()

            all_probs.extend(myopia_probs)
            all_targets.extend(true_labels)

    metrics = compute_classification_metrics(all_targets, all_probs, threshold=0.5)
    return metrics, all_targets, all_probs

# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_t2_fundus(args):
    # 1. Hardware & Lingkungan Setup
    print_hardware_summary()
    device = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    print(f"[Device]: Menggunakan {device.type.upper()}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  -> cuDNN Benchmark diaktifkan untuk throughput konvolusi maksimal pada RTX 5050.")

    # 2. Direktori Run
    run_dir = Path(f"ml/runs/fundus_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run Folder]: {run_dir.resolve()}")

    # 3. Load / Buat Split Data
    split_file = Path("ml/train/split_fundus.json")
    if split_file.exists():
        print(f"[Data]: Memuat split dari {split_file}...")
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_rows = split_data["train"]
        val_rows = split_data["val"]
        test_rows = split_data["test"]
    else:
        print("[Data]: split_fundus.json belum ada. Memindai folder Myopia Images...")
        all_rows = build_myopia_index()
        train_rows, val_rows, test_rows = stratified_myopia_split(all_rows, seed=SEED)
        with open(split_file, "w") as f:
            json.dump({"train": train_rows, "val": val_rows, "test": test_rows}, f)
        print(f"  -> Split tersimpan ke {split_file}")

    # VERIFIKASI INTEGRITAS SPLIT
    tr_set = {r["img"] for r in train_rows}
    va_set = {r["img"] for r in val_rows}
    te_set = {r["img"] for r in test_rows}
    assert not (tr_set & va_set), "LEAKAGE: Ada file sama di Train dan Val!"
    assert not (tr_set & te_set), "LEAKAGE: Ada file sama di Train dan Test!"
    assert not (va_set & te_set), "LEAKAGE: Ada file sama di Val dan Test!"

    print("[Anti-Leakage Check]: LULUS! Irisan file train-val-test = 0 ✅")
    print(f"  Train : {len(train_rows)} gambar ({sum(r['label'] for r in train_rows)} Myopia)")
    print(f"  Val   : {len(val_rows)} gambar ({sum(r['label'] for r in val_rows)} Myopia)")
    print(f"  Test  : {len(test_rows)} gambar ({sum(r['label'] for r in test_rows)} Myopia)")

    # 4. Sanity Check (32 Sampel)
    if not args.skip_sanity:
        sanity_passed = run_fundus_sanity_check(train_rows, device=device, use_amp=args.amp)
        if args.only_sanity:
            print("Mode --only-sanity aktif. Selesai.")
            return
        if not sanity_passed and not args.force:
            print("Error: Sanity check gagal. Batalkan training penuh.")
            return

    # 5. DataLoaders dengan Konfigurasi Optimal RTX 5050 & i7
    use_workers = args.num_workers if os.name != "nt" or args.num_workers <= 8 else 8
    print(f"[DataLoader Config]: batch_size={args.batch_size}, num_workers={use_workers}, pin_memory=True")

    train_ds = FundusDataset(train_rows, augment=True)
    val_ds = FundusDataset(val_rows, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(use_workers > 0),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=max(2, use_workers // 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(use_workers > 0)
    )

    # 6. Model, Optimizer, Scheduler, Loss & AMP Scaler
    model = build_fundus_model(num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("[AMP]: Automatic Mixed Precision (FP16 Tensor Cores) AKTIF 🚀")

    # 7. Training Loop Utama
    best_val_auc = 0.0
    patience = args.patience
    bad_epochs = 0
    history = []

    print("\n" + "=" * 75)
    print(f" 🚀 MEMULAI TRAINING MYOPIA FUNDUS CLASSIFIER ({args.epochs} EPOCHS)")
    print(f" Target JagaMata: ROC-AUC >= 0.90 | Early Stopping Patience: {patience}")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y_onehot in train_loader:
            x = x.to(device, non_blocking=True)
            y_onehot = y_onehot.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = loss_fn(logits, y_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y_onehot)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(x)
            n_samples += len(x)

        train_loss = running_loss / max(n_samples, 1)

        # Evaluasi Validasi
        val_metrics, _, _ = evaluate_classification(model, val_loader, device=device, use_amp=args.amp)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]
        val_sens = val_metrics["sensitivity"]
        val_spec = val_metrics["specificity"]

        scheduler.step()
        epoch_sec = round(time.time() - t0, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_sensitivity": val_sens,
            "val_specificity": val_spec,
            "lr": float(f"{current_lr:.6f}"),
            "duration_sec": epoch_sec
        }
        history.append(epoch_record)

        print(
            f" Epoch {epoch:02d}/{args.epochs:02d} | Train BCE: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Sens: {val_sens:.3f} | Spec: {val_spec:.3f} | "
            f"{epoch_sec}s | Best AUC: {best_val_auc:.4f}"
        )

        # Simpan checkpoint model terbaik (berdasarkan Val AUC)
        if val_auc > best_val_auc + 0.002:
            best_val_auc = val_auc
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"   -> [BEST MODEL SAVED] Val AUC baru: {best_val_auc:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\n[Early Stopping]: Tidak ada peningkatan selama {patience} epoch.")
                break

    # Simpan metrics histori lengkap
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # 8. TEST FINAL PADA TEST SET (SEKALI SAJA DI AKHIR)
    print("\n" + "=" * 75)
    print(" 🎯 EVALUASI TEST FINAL MYOPIA FUNDUS (SEKALI PAKAI)")
    print("=" * 75)

    best_weights_path = run_dir / "best.pt"
    if not best_weights_path.exists():
        torch.save(model.state_dict(), best_weights_path)

    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_ds = FundusDataset(test_rows, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=max(2, use_workers // 2)
    )

    test_metrics, test_tgts, test_probs = evaluate_classification(
        model, test_loader, device=device, use_amp=args.amp
    )
    ci_low, ci_high = bootstrap_auc_ci95(test_tgts, test_probs, n_resamples=1000, seed=SEED)

    kpi_target = 0.90
    kpi_pass = bool(test_metrics["auc"] >= kpi_target)

    test_results = {
        "task": "T2_Myopia_Fundus_Classifier",
        "model": "EfficientNet-B0",
        "n_test_samples": len(test_rows),
        "test_auc": test_metrics["auc"],
        "ci95_auc": [round(ci_low, 4), round(ci_high, 4)],
        "test_f1": test_metrics["f1"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "test_accuracy": test_metrics["accuracy"],
        "confusion_matrix": {
            "tp": test_metrics["tp"], "tn": test_metrics["tn"],
            "fp": test_metrics["fp"], "fn": test_metrics["fn"]
        },
        "val_auc_best": round(best_val_auc, 4),
        "target_kpi": f"AUC >= {kpi_target}",
        "lolos_target": kpi_pass,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    test_res_file = run_dir / "test_results.json"
    with open(test_res_file, "w") as f:
        json.dump(test_results, f, indent=2)

    print(json.dumps(test_results, indent=2))
    print(f"\nHasil Test Disimpan -> {test_res_file}")
    print(f"Histori Metrics Disimpan -> {run_dir / 'metrics.json'}")

    # 9. Plot Learning Curves jika matplotlib tersedia
    try:
        import matplotlib.pyplot as plt
        epochs_range = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_aucs = [h["val_auc"] for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train BCE Loss", color=color)
        ax1.plot(epochs_range, train_losses, label="Train Loss", color=color, linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Val AUC", color=color)
        ax2.plot(epochs_range, val_aucs, label="Val AUC", color=color, linewidth=2, linestyle="--")
        ax2.axhline(y=0.90, color="red", linestyle=":", label="Target KPI (0.90)")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Learning Curves - T2 Myopia Fundus (Best Val AUC: {best_val_auc:.4f})")
        fig.tight_layout()
        curves_path = run_dir / "curves.png"
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Plot Kurva Training -> {curves_path}")
    except Exception as e:
        print(f"(Catatan plot dilewati: {e})")

    print("\n✅ Training T2 Selesai dengan Sukses!")

# =============================================================================
# CLI PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="JagaMata Task 2: Myopia Fundus Training")
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch training (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Ukuran batch (default: 64, optimal untuk 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay AdamW (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader num_workers (default: 8, optimal i7)")
    parser.add_argument("--patience", type=int, default=10, help="Patience early stopping (default: 10)")
    parser.add_argument("--device", type=str, default="cuda", help="Perangkat komputasi: 'cuda' atau 'cpu'")
    parser.add_argument("--amp", action="store_true", default=True, help="Aktifkan Automatic Mixed Precision")
    parser.add_argument("--skip-sanity", action="store_true", help="Lewati sanity check awal")
    parser.add_argument("--only-sanity", action="store_true", help="Hanya jalankan sanity check lalu keluar")
    parser.add_argument("--force", action="store_true", help="Paksa lanjut walau sanity check gagal")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_t2_fundus(args)
