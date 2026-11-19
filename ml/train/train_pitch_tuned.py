"""
=============================================================================
JagaMata ML Training Suite - Task 1 (T1) Tuned: Anti-Oversmoothing Pitch Regressor
=============================================================================
Penyempurnaan AI Berdasarkan Analisis Baseline & Panduan §6 Eksperimen #2:
1. Multi-Task Joint Pose Regression: Model memprediksi 3 sudut [pitch, yaw, roll]
   bersama-sama. Menghilangkan ambiguitas rotasi 3D dan mencegah "oversmoothing"
   pada sudut yaw ekstrem. Evaluasi & early stopping tetap diprioritaskan pada PITCH.
2. Regularisasi Anti-Overfitting:
   - Dropout 0.3 pada classifier head
   - Weight decay ditingkatkan dari 1e-4 -> 1e-3
3. Loss Function: SmoothL1Loss (beta=1.5) untuk gradien yang lebih responsif
4. Evaluasi Komprehensif: Menghitung MAE Pitch keseluruhan AFLW2000-3D dan
   subgroup operasional JagaMata (|yaw| < 45°, rentang anak menatap layar HP)
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

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# DATASET DENGAN 3 SUDUT POSE [PITCH, YAW, ROLL]
# =============================================================================

class FullPoseDataset(Dataset):
    """
    Dataset loader yang memuat 3 sudut pose (pitch, yaw, roll dalam derajat).
    Horizontal Flip disesuaikan secara matematis:
    - Pitch: TETAP (tidak berubah saat dicerminkan secara horizontal)
    - Yaw: DIBALIK tandanya (-yaw)
    - Roll: DIBALIK tandanya (-roll)
    """
    def __init__(self, rows: List[Dict[str, str]], augment: bool = False):
        self.rows = rows
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
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
                
                # Flip horizontal manual untuk sinkronisasi tanda sudut
                do_flip = self.augment and (random.random() < 0.5)
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                tensor_img = self.transform(img)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            do_flip = False

        pitch, yaw, roll = read_pose_mat(mat_path)
        if do_flip:
            yaw = -yaw
            roll = -roll

        # Target 3D pose: [pitch, yaw, roll]
        targets = torch.tensor([pitch, yaw, roll], dtype=torch.float32)
        return tensor_img, targets

# =============================================================================
# ARSITEKTUR MODEL TUNED DENGAN DROPOUT & 3 OUTPUTS
# =============================================================================

def build_tuned_pitch_model(pretrained: bool = True, dropout_p: float = 0.3) -> nn.Module:
    """
    MobileNetV3-Small dengan classifier head yang diperkuat regularisasi:
    Linear(in, 1024) -> Hardswish -> Dropout(dropout_p) -> Linear(1024, 3)
    Output: [pitch, yaw, roll]
    """
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(pretrained=pretrained)

    in_features = model.classifier[3].in_features
    # Ganti head classifier dengan regularisasi dropout dan output 3 pose
    model.classifier[2] = nn.Dropout(p=dropout_p)
    model.classifier[3] = nn.Linear(in_features, 3)
    return model

# =============================================================================
# EVALUASI METRIK PITCH & SUBGROUP
# =============================================================================

def evaluate_pose(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, Any]:
    model.eval()
    all_pred_pitch = []
    all_tgt_pitch = []
    all_pred_yaw = []
    all_tgt_yaw = []
    all_pred_roll = []
    all_tgt_roll = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x)
            else:
                out = model(x)

            all_pred_pitch.extend(out[:, 0].cpu().tolist())
            all_tgt_pitch.extend(y[:, 0].tolist())
            all_pred_yaw.extend(out[:, 1].cpu().tolist())
            all_tgt_yaw.extend(y[:, 1].tolist())
            all_pred_roll.extend(out[:, 2].cpu().tolist())
            all_tgt_roll.extend(y[:, 2].tolist())

    import numpy as np
    pred_p = np.array(all_pred_pitch)
    tgt_p = np.array(all_tgt_pitch)
    tgt_y = np.array(all_tgt_yaw)

    # MAE pitch keseluruhan
    mae_pitch_overall = float(np.mean(np.abs(pred_p - tgt_p)))

    # MAE pitch pada rentang tatap layar operasional JagaMata (|yaw| < 45°)
    mask_screen = np.abs(tgt_y) < 45.0
    mae_pitch_screen = float(np.mean(np.abs(pred_p[mask_screen] - tgt_p[mask_screen]))) if np.sum(mask_screen) > 0 else mae_pitch_overall
    n_screen = int(np.sum(mask_screen))

    # MAE pitch pada sudut ekstrem (|yaw| >= 45°)
    mask_extreme = ~mask_screen
    mae_pitch_extreme = float(np.mean(np.abs(pred_p[mask_extreme] - tgt_p[mask_extreme]))) if np.sum(mask_extreme) > 0 else 0.0
    n_extreme = int(np.sum(mask_extreme))

    return {
        "mae_pitch_overall": round(mae_pitch_overall, 4),
        "mae_pitch_screen": round(mae_pitch_screen, 4),
        "mae_pitch_extreme": round(mae_pitch_extreme, 4),
        "n_samples": len(pred_p),
        "n_screen_samples": n_screen,
        "n_extreme_samples": n_extreme,
        "pred_pitch": pred_p,
        "tgt_pitch": tgt_p,
    }

# =============================================================================
# SANITY CHECK (32 SAMPEL)
# =============================================================================

def run_tuned_sanity_check(train_rows: List[Dict[str, str]], device: torch.device, use_amp: bool = True) -> bool:
    print("\n" + "=" * 65)
    print(" 🧪 MENJALANKAN SANITY CHECK (TUNED POSE REGRESSOR: 32 SAMPEL)")
    print("=" * 65)

    sample_rows = train_rows[:32]
    dataset = FullPoseDataset(sample_rows, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_tuned_pitch_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=1.5)

    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    initial_loss = None
    final_loss = None

    model.train()
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
        if ep == 1: initial_loss = avg_loss
        final_loss = avg_loss

        if ep % 5 == 0 or ep == 25:
            print(f"   [Sanity Epoch {ep:02d}/25] Loss: {avg_loss:.4f}", flush=True)

    ratio = final_loss / (initial_loss + 1e-7)
    print(f"   Loss Awal: {initial_loss:.4f} -> Loss Akhir: {final_loss:.4f} (Rasio: {ratio:.1%})")
    passed = ratio < 0.15
    if passed:
        print("   ✅ SANITY CHECK TUNED LULUS! Pipeline normal & gradient lancar.")
    else:
        print("   ❌ SANITY CHECK GAGAL!")
    print("=" * 65 + "\n")
    return passed

# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_t1_pitch_tuned(args):
    print_hardware_summary()
    device = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    print(f"[Device]: Menggunakan {device.type.upper()}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  -> cuDNN Benchmark diaktifkan untuk RTX 5050.")

    if hasattr(args, "run_name") and args.run_name:
        run_dir = Path(f"ml/runs/{args.run_name}")
    else:
        run_dir = Path(f"ml/runs/pitch_tuned_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run Folder]: {run_dir.resolve()}")

    split_file = Path("ml/train/split_pitch.json")
    with open(split_file, "r") as f:
        split_data = json.load(f)
    train_rows = split_data["train"]
    val_rows = split_data["val"]
    test_rows = split_data["test"]

    print(f"[Data]: Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    if not args.skip_sanity:
        run_tuned_sanity_check(train_rows, device=device, use_amp=args.amp)
        if args.only_sanity:
            return

    use_workers = args.num_workers if os.name != "nt" or args.num_workers <= 8 else 8
    print(f"[DataLoader Config]: batch_size={args.batch_size}, num_workers={use_workers}, pin_memory=True")

    train_ds = FullPoseDataset(train_rows, augment=True)
    val_ds = FullPoseDataset(val_rows, augment=False)

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

    dropout_p = getattr(args, "dropout", 0.3)
    model = build_tuned_pitch_model(pretrained=True, dropout_p=dropout_p).to(device)
    # Eksperimen §6: AdamW + Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_ep = getattr(args, "warmup_epochs", 0)
    if warmup_ep > 0:
        w_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_ep)
        c_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [w_sched, c_sched], milestones=[warmup_ep])
        print(f"[Scheduler]: Linear Warmup ({warmup_ep} epochs) -> Cosine Annealing (eta_min=1e-6)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"[Scheduler]: Cosine Annealing (T_max={args.epochs}, eta_min=1e-6)")

    huber_beta = getattr(args, "beta", 1.5)
    loss_fn = nn.SmoothL1Loss(beta=huber_beta)

    pitch_wt = getattr(args, "pitch_weight", 1.5)
    loss_weights = torch.tensor([pitch_wt, 1.0, 1.0], device=device)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    if scaler:
        print("[AMP]: FP16 Tensor Cores AKTIF 🚀")

    best_val_pitch_mae = math.inf
    patience = args.patience
    bad_epochs = 0
    history = []

    print("\n" + "=" * 80)
    print(f" 🚀 MEMULAI TRAINING TUNED HEAD PITCH REGRESSOR ({args.epochs} EPOCHS)")
    print(f" Setting: LR: {args.lr} | Dropout: {dropout_p} | Weight Decay: {args.weight_decay} | Beta: {huber_beta} | Pitch Weight: {pitch_wt}x")
    print(f" Target: MAE Pitch Operasional Screen (|yaw| < 45°) <= 5.0°")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0

        total_batches = len(train_loader)
        for b_idx, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    preds = model(x)
                    loss = loss_fn(preds * loss_weights, y * loss_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x)
                loss = loss_fn(preds * loss_weights, y * loss_weights)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(x)
            n_samples += len(x)

            if b_idx % 200 == 0:
                cur_loss = running_loss / max(n_samples, 1)
                print(f"    [Ep {epoch:02d}] Batch {b_idx:03d}/{total_batches} ({n_samples}/{len(train_ds)}) | Huber: {cur_loss:.4f}", flush=True)

        train_loss = running_loss / max(n_samples, 1)

        val_res = evaluate_pose(model, val_loader, device=device, use_amp=args.amp)
        v_pitch_mae = val_res["mae_pitch_overall"]
        v_pitch_screen = val_res["mae_pitch_screen"]

        scheduler.step()
        epoch_sec = round(time.time() - t0, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_pitch_mae": v_pitch_mae,
            "val_pitch_screen_mae": v_pitch_screen,
            "lr": float(f"{current_lr:.6f}"),
            "duration_sec": epoch_sec
        }
        history.append(epoch_record)

        print(
            f" Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Pitch MAE: {v_pitch_mae:.2f}° (Screen: {v_pitch_screen:.2f}°) | "
            f"{epoch_sec}s | Best: {best_val_pitch_mae:.2f}°",
            flush=True
        )

        if v_pitch_mae < best_val_pitch_mae - 0.01:
            best_val_pitch_mae = v_pitch_mae
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"   -> [BEST MODEL SAVED] Val Pitch MAE baru: {best_val_pitch_mae:.2f}°", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\n[Early Stopping]: Selesai pada epoch {epoch}.", flush=True)
                break

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # =========================================================================
    # TEST FINAL SAKRAL PADA AFLW2000-3D
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 🎯 EVALUASI TEST FINAL TUNED: AFLW2000-3D (SEKALI PAKAI)")
    print("=" * 80)

    best_weights_path = run_dir / "best.pt"
    if not best_weights_path.exists():
        torch.save(model.state_dict(), best_weights_path)

    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_ds = FullPoseDataset(test_rows, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=max(2, use_workers // 2)
    )

    test_res = evaluate_pose(model, test_loader, device=device, use_amp=args.amp)
    ci_low, ci_high = bootstrap_mae_ci95(test_res["pred_pitch"], test_res["tgt_pitch"], n_resamples=1000, seed=SEED)

    kpi_target = 5.0
    kpi_pass_screen = bool(test_res["mae_pitch_screen"] <= kpi_target)

    test_results = {
        "task": "T1_Head_Pitch_Regressor_Tuned",
        "model": "MobileNetV3-Small (Multi-Task + Dropout 0.3)",
        "test_dataset": "AFLW2000-3D",
        "n_test_samples": len(test_rows),
        "test_mae_pitch_overall": test_res["mae_pitch_overall"],
        "test_mae_pitch_screen_domain": test_res["mae_pitch_screen"],
        "test_mae_pitch_extreme_yaw": test_res["mae_pitch_extreme"],
        "ci95_overall": [round(ci_low, 4), round(ci_high, 4)],
        "val_mae_pitch_best": round(best_val_pitch_mae, 4),
        "target_kpi": f"MAE pitch <= {kpi_target}°",
        "lolos_target_screen_domain": kpi_pass_screen,
        "n_screen_domain_samples": test_res["n_screen_samples"],
        "n_extreme_yaw_samples": test_res["n_extreme_samples"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    test_res_file = run_dir / "test_results.json"
    with open(test_res_file, "w") as f:
        json.dump(test_results, f, indent=2)

    print(json.dumps(test_results, indent=2))
    print(f"\nHasil Test Disimpan -> {test_res_file}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        ep_range = [h["epoch"] for h in history]
        tr_loss = [h["train_loss"] for h in history]
        va_pitch = [h["val_pitch_mae"] for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Huber Loss", color=color)
        ax1.plot(ep_range, tr_loss, label="Train Loss", color=color, linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("Val Pitch MAE (°)", color=color)
        ax2.plot(ep_range, va_pitch, label="Val Pitch MAE", color=color, linewidth=2, linestyle="--")
        ax2.axhline(y=5.0, color="green", linestyle=":", label="KPI Target (5.0°)")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Tuned Learning Curves - T1 Pitch (Best Val MAE: {best_val_pitch_mae:.2f}°)")
        fig.tight_layout()
        plt.savefig(run_dir / "curves.png", dpi=150)
        plt.close()
        print(f"Plot Kurva Disimpan -> {run_dir / 'curves.png'}")
    except Exception as e:
        print(f"(Catatan plot: {e})")

    print("\n✅ Training Tuned T1 Selesai dengan Sukses!")

def parse_args():
    parser = argparse.ArgumentParser(description="JagaMata Task 1 Tuned: Anti-Oversmoothing Pitch Training")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch training (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Ukuran batch (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay AdamW (default: 1e-3, regularisasi anti-overfitting)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability pada classifier head (default: 0.3)")
    parser.add_argument("--beta", type=float, default=1.5, help="Huber beta threshold untuk SmoothL1Loss (default: 1.5)")
    parser.add_argument("--pitch-weight", type=float, default=1.5, help="Bobot loss multiplier untuk sudut pitch (default: 1.5)")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Jumlah epoch linear warmup (default: 0)")
    parser.add_argument("--run-name", type=str, default=None, help="Nama folder kustom untuk menyimpan artefak di ml/runs/")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader num_workers (default: 8)")
    parser.add_argument("--patience", type=int, default=15, help="Patience early stopping (default: 15)")
    parser.add_argument("--device", type=str, default="cuda", help="Perangkat komputasi: 'cuda' atau 'cpu'")
    parser.add_argument("--amp", action="store_true", default=True, help="Aktifkan Automatic Mixed Precision")
    parser.add_argument("--skip-sanity", action="store_true", help="Lewati sanity check awal")
    parser.add_argument("--only-sanity", action="store_true", help="Hanya jalankan sanity check")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_t1_pitch_tuned(args)
