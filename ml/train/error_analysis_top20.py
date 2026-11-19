import json
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_pitch_tuned import build_tuned_pitch_model, FullPoseDataset
from common import read_pose_mat

def main():
    split_file = Path("ml/train/split_pitch.json")
    with open(split_file, "r") as f:
        test_rows = json.load(f)["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("ml/runs/pitch_exp_pitchweight2_beta1/best.pt")
    if not model_path.exists():
        model_path = Path("ml/runs/pitch_tuned_20261118_210456/best.pt")
    print(f"Loading weights from: {model_path} on {device}")
    model = build_tuned_pitch_model(pretrained=False, dropout_p=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = FullPoseDataset(test_rows, augment=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    results = []
    curr_idx = 0
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(test_loader):
            print(f"Evaluasi batch {b_idx+1}/{len(test_loader)}...", flush=True)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x.to(device, non_blocking=True))
            else:
                out = model(x.to(device))
            preds = out.cpu().tolist()
            trues = y.tolist()
            for p, t in zip(preds, trues):
                r = test_rows[curr_idx]
                curr_idx += 1
                pred_p, pred_y, pred_r = p
                true_p, true_y, true_r = t
                err_pitch = abs(pred_p - true_p)
                results.append({
                    "filename": Path(r["jpg"]).name,
                    "true_pitch": round(true_p, 2),
                    "pred_pitch": round(pred_p, 2),
                    "pitch_error": round(err_pitch, 2),
                    "true_yaw": round(true_y, 2),
                    "pred_yaw": round(pred_y, 2),
                    "true_roll": round(true_r, 2),
                    "pred_roll": round(pred_r, 2),
                    "is_screen_domain": abs(true_y) < 45.0
                })

    # Urutkan berdasarkan error pitch terbesar
    results.sort(key=lambda x: x["pitch_error"], reverse=True)
    top20 = results[:20]

    out_file = Path("ml/runs/top20_worst_errors.json")
    with open(out_file, "w") as f:
        json.dump(top20, f, indent=2)

    print("\n" + "=" * 85)
    print(" 🔴 TOP 20 SAMPEL ERROR PITCH TERBURUK (AFLW2000-3D)")
    print("=" * 85)
    print(f"{'Filename':<18} | {'True P':<7} | {'Pred P':<7} | {'Error':<7} | {'True Yaw':<8} | {'True Roll':<9} | {'Domain':<12}")
    print("-" * 85)
    for s in top20:
        dom = "Screen" if s["is_screen_domain"] else "Extreme Yaw"
        print(f"{s['filename']:<18} | {s['true_pitch']:>6.1f}° | {s['pred_pitch']:>6.1f}° | {s['pitch_error']:>6.1f}° | {s['true_yaw']:>7.1f}° | {s['true_roll']:>8.1f}° | {dom:<12}")
    print("=" * 85 + "\n")

    # Statistik karakteristik top 20
    extreme_count = sum(1 for s in top20 if not s["is_screen_domain"])
    print(f"Proporsi sampel dengan |yaw| >= 45° di Top 20: {extreme_count}/20 ({extreme_count*5}%)")

if __name__ == "__main__":
    main()
