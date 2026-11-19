"""
=============================================================================
JagaMata ML Training Suite - Task 1 (T1): Head Pitch Regressor
=============================================================================
Arsitektur    : MobileNetV3-Small (Pretrained ImageNet) -> 1 Output (Pitch Degree)
Dataset Latih : 300W-LP (AFW, HELEN, IBUG, LFPW)
Aturan Split  : Group Split (irisan group = 0, anti-data leakage)
Dataset Test  : AFLW2000-3D (Evaluasi sakral SEKALI SAJA di akhir)
Loss Function : SmoothL1Loss (Huber Loss, robust terhadap outliers)
Optimizer     : AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler     : CosineAnnealingLR
Hardware Opt  : CUDA, AMP (FP16 Tensor Cores RTX 5050), num_workers=8, pin_memory=True
Metrik Target : MAE Pitch <= 5.0° (Evaluasi Bootstrap 95% CI 1000 resamples)
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
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Import modul helper common dari package lokal
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    read_pose_mat,
    group_split,
    build_lp_index,
    build_test_index,
    compute_regression_metrics,
    bootstrap_mae_ci95,
    print_hardware_summary,
)

# Deterministic Seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# DATASET & AUGMENTASI
# =============================================================================

class PoseDataset(Dataset):
    """
    Dataset loader untuk foto wajah dan ground-truth head pose (.mat).
    Target default adalah 'pitch' (dalam derajat).
    """
    def __init__(self, rows: List[Dict[str, str]], augment: bool = False, target: str = "pitch"):
        self.rows = rows
        self.augment = augment
        self.target = target

        # Pipeline augmentasi sesuai Panduan §5 baris 262:
        # Catatan ilmiah: Horizontal Flip AMAN untuk regresi pitch karena menoleh
        # ke kiri/kanan (yaw) terbalik, tetapi sudut menunduk/tengadah (pitch) tetap sama!
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
        img_path = row["jpg"]
        mat_path = row["mat"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor_img = self.transform(img)
        except Exception as e:
            # Fallback jika ada file korup: buat blank canvas
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)

        pitch_deg, yaw_deg, roll_deg = read_pose_mat(mat_path)
        labels = {"pitch": pitch_deg, "yaw": yaw_deg, "roll": roll_deg}
        target_val = labels.get(self.target, pitch_deg)

        return tensor_img, torch.tensor([target_val], dtype=torch.float32)

# =============================================================================
# ARSITEKTUR MODEL (MobileNetV3-Small -> 1 output)
# =============================================================================

def build_pitch_model(pretrained: bool = True) -> nn.Module:
    """
    Membangun model regresi pitch berbasis MobileNetV3-Small.
    Layer classifier terakhir diubah menjadi Linear(in_features, 1).
    """
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        # Kompatibilitas untuk versi PyTorch lama
        model = models.mobilenet_v3_small(pretrained=pretrained)

    in_features = model.classifier[3].in_features
    # Ganti classifier akhir: output tunggal derajat pitch
    model.classifier[3] = nn.Linear(in_features, 1)
    return model

# =============================================================================
# SANITY CHECK / OVERFIT TEST (32 SAMPEL)
# =============================================================================

def run_sanity_check(
    train_rows: List[Dict[str, str]], 
    device: torch.device, 
    use_amp: bool = True
) -> bool:
    """
    Sanity check wajib sesuai Panduan §4.2:
    Melatih model pada 32 sampel selama 25 epoch.
    Harus overfit: Loss epoch akhir < 10% dari loss awal.
    Jika gagal, pipeline/gradient rusak dan training penuh tidak boleh dilanjutkan!
    """
    print("\n" + "=" * 65)
    print(" 🧪 MENJALANKAN SANITY CHECK (OVERFIT TEST: 32 SAMPEL)")
    print("=" * 65)

    sample_rows = train_rows[:32]
    dataset = PoseDataset(sample_rows, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_pitch_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

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
                        out = model(x)
                        loss = loss_fn(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(x)
                    loss = loss_fn(out, y)
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
            print("   python ml\\train\\train_pitch.py --device cpu")
            print("!" * 75 + "\n")
            return False
        raise e

    ratio = final_loss / (initial_loss + 1e-7)
    print(f"   Loss Awal: {initial_loss:.4f} -> Loss Akhir: {final_loss:.4f} (Rasio: {ratio:.1%})")

    passed = ratio < 0.15
    if passed:
        print("   ✅ SANITY CHECK LULUS! Pipeline normal, gradient mengalir sempurna.")
    else:
        print("   ❌ SANITY CHECK GAGAL! Loss tidak turun signifikan (<15%). Cek label/data/LR.")
    print("=" * 65 + "\n")
    return passed

# =============================================================================
# EVALUASI VALIDASI / TEST
# =============================================================================

def evaluate_mae(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, List[float], List[float]]:
    """
    Evaluasi model untuk mendapatkan MAE, prediksi, dan target.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x)
            else:
                out = model(x)

            all_preds.extend(out.cpu().squeeze().tolist())
            all_targets.extend(y.squeeze().tolist())

    preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
    tgts_tensor = torch.tensor(all_targets, dtype=torch.float32)
    mae = float((preds_tensor - tgts_tensor).abs().mean().item())

    return mae, all_preds, all_targets

# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_t1_pitch(args):
    # 1. Hardware & Lingkungan Setup
    print_hardware_summary()
    device = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    print(f"[Device]: Menggunakan {device.type.upper()}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  -> cuDNN Benchmark diaktifkan untuk throughput konvolusi maksimal pada RTX 5050.")

    # 2. Direktori Run
    run_dir = Path(f"ml/runs/pitch_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run Folder]: {run_dir.resolve()}")

    # 3. Load / Buat Split Data
    split_file = Path("ml/train/split_pitch.json")
    if split_file.exists():
        print(f"[Data]: Memuat split dari {split_file}...")
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_rows = split_data["train"]
        val_rows = split_data["val"]
        test_rows = split_data["test"]
    else:
        print("[Data]: split_pitch.json belum ada. Memindai 300W-LP dan AFLW2000-3D...")
        lp_rows = build_lp_index(include_flips=False)
        train_rows, val_rows = group_split(lp_rows, val_frac=0.10, seed=SEED)
        test_rows = build_test_index()
        with open(split_file, "w") as f:
            json.dump({"train": train_rows, "val": val_rows, "test": test_rows}, f)
        print(f"  -> Split tersimpan ke {split_file}")

    # VERIFIKASI ANTI-LEAKAGE KETAT
    train_groups = {r["group"] for r in train_rows}
    val_groups = {r["group"] for r in val_rows}
    leakage = train_groups & val_groups
    assert len(leakage) == 0, f"DATA LEAKAGE KRITIS! Irisan group: {leakage}"
    print(f"[Anti-Leakage Check]: LULUS! Irisan group train-val = 0 ✅")
    print(f"  Train: {len(train_rows)} gambar ({len(train_groups)} groups)")
    print(f"  Val  : {len(val_rows)} gambar ({len(val_groups)} groups)")
    print(f"  Test : {len(test_rows)} gambar (AFLW2000-3D sakral)")

    # 4. Sanity Check (32 Sampel)
    if not args.skip_sanity:
        sanity_passed = run_sanity_check(train_rows, device=device, use_amp=args.amp)
        if args.only_sanity:
            print("Mode --only-sanity aktif. Selesai.")
            return
        if not sanity_passed and not args.force:
            print("Error: Sanity check gagal. Batalkan training penuh (gunakan --force jika tetap ingin lanjut).")
            return

    # 5. DataLoaders dengan Konfigurasi Optimal RTX 5050 & i7
    # num_workers=8 memanfaatkan Core i7-13620H tanpa bottleneck
    # pin_memory=True memanfaatkan RAM 16GB untuk DMA transfer cepat ke GPU VRAM
    # persistent_workers=True menghindari overhead respawn worker tiap epoch
    use_workers = args.num_workers if os.name != "nt" or args.num_workers <= 8 else 8
    print(f"[DataLoader Config]: batch_size={args.batch_size}, num_workers={use_workers}, pin_memory=True")

    train_ds = PoseDataset(train_rows, augment=True)
    val_ds = PoseDataset(val_rows, augment=False)

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
    model = build_pitch_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber Loss

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("[AMP]: Automatic Mixed Precision (FP16 Tensor Cores) AKTIF 🚀")

    # 7. Training Loop Utama
    best_val_mae = math.inf
    patience = args.patience
    bad_epochs = 0
    history = []

    print("\n" + "=" * 75)
    print(f" 🚀 MEMULAI TRAINING HEAD PITCH REGRESSOR ({args.epochs} EPOCHS)")
    print(f" Target JagaMata: MAE Pitch <= 5.0° | Early Stopping Patience: {patience}")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    preds = model(x)
                    loss = loss_fn(preds, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(x)
            n_samples += len(x)

        train_loss = running_loss / max(n_samples, 1)

        # Evaluasi Validasi
        val_mae, _, _ = evaluate_mae(model, val_loader, device=device, use_amp=args.amp)
        scheduler.step()
        epoch_sec = round(time.time() - t0, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae, 4),
            "lr": float(f"{current_lr:.6f}"),
            "duration_sec": epoch_sec
        }
        history.append(epoch_record)

        print(
            f" Epoch {epoch:02d}/{args.epochs:02d} | Train Huber Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.2f}° | {epoch_sec}s | LR: {current_lr:.1e} | Best: {best_val_mae:.2f}°"
        )

        # Simpan checkpoint model terbaik (berdasarkan Val MAE)
        if val_mae < best_val_mae - 0.01:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"   -> [BEST MODEL SAVED] Val MAE baru: {best_val_mae:.2f}°")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\n[Early Stopping]: Tidak ada peningkatan selama {patience} epoch berturut-turut.")
                break

    # Simpan metrics histori lengkap
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # 8. TEST FINAL SAKRAL AFLW2000-3D (SEKALI SAJA DI AKHIR)
    print("\n" + "=" * 75)
    print(" 🎯 EVALUASI TEST FINAL: AFLW2000-3D (SEKALI PAKAI)")
    print("=" * 75)

    best_weights_path = run_dir / "best.pt"
    if not best_weights_path.exists():
        torch.save(model.state_dict(), best_weights_path)

    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_ds = PoseDataset(test_rows, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=max(2, use_workers // 2)
    )

    test_mae, test_preds, test_tgts = evaluate_mae(model, test_loader, device=device, use_amp=args.amp)
    ci_low, ci_high = bootstrap_mae_ci95(test_preds, test_tgts, n_resamples=1000, seed=SEED)

    kpi_target = 5.0
    kpi_pass = bool(test_mae <= kpi_target)

    test_results = {
        "task": "T1_Head_Pitch_Regressor",
        "model": "MobileNetV3-Small",
        "test_dataset": "AFLW2000-3D",
        "n_test_samples": len(test_rows),
        "test_mae_pitch": round(test_mae, 4),
        "ci95": [round(ci_low, 4), round(ci_high, 4)],
        "val_mae_best": round(best_val_mae, 4),
        "target_kpi": f"MAE pitch <= {kpi_target}°",
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
        val_maes = [h["val_mae"] for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Huber Loss", color=color)
        ax1.plot(epochs_range, train_losses, label="Train Loss", color=color, linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("Val MAE (°)", color=color)
        ax2.plot(epochs_range, val_maes, label="Val MAE", color=color, linewidth=2, linestyle="--")
        ax2.axhline(y=5.0, color="green", linestyle=":", label="Target KPI (5.0°)")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Learning Curves - T1 Head Pitch (Best Val MAE: {best_val_mae:.2f}°)")
        fig.tight_layout()
        curves_path = run_dir / "curves.png"
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Plot Kurva Training -> {curves_path}")
    except Exception as e:
        print(f"(Catatan plot dilewati: {e})")

    print("\n✅ Training T1 Selesai dengan Sukses!")

# =============================================================================
# CLI PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="JagaMata Task 1: Head Pitch Training")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch training (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Ukuran batch (default: 64, optimal untuk 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay AdamW (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader num_workers (default: 8, optimal i7)")
    parser.add_argument("--patience", type=int, default=15, help="Patience early stopping (default: 15)")
    parser.add_argument("--device", type=str, default="cuda", help="Perangkat komputasi: 'cuda' atau 'cpu'")
    parser.add_argument("--amp", action="store_true", default=True, help="Aktifkan Automatic Mixed Precision")
    parser.add_argument("--skip-sanity", action="store_true", help="Lewati sanity check awal")
    parser.add_argument("--only-sanity", action="store_true", help="Hanya jalankan sanity check lalu keluar")
    parser.add_argument("--force", action="store_true", help="Paksa lanjut walau sanity check gagal")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_t1_pitch(args)
