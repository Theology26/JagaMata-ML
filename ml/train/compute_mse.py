import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pitch import build_pitch_model, PoseDataset
from train_pitch_tuned import build_tuned_pitch_model, FullPoseDataset
from common import read_pose_mat

def evaluate_model_mse(model_type, checkpoint_path, test_rows, device):
    loader = DataLoader(
        FullPoseDataset(test_rows, augment=False) if model_type == "tuned" else PoseDataset(test_rows, augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    if model_type == "baseline":
        model = build_pitch_model(pretrained=False).to(device)
    else:
        model = build_tuned_pitch_model(pretrained=False, dropout_p=0.3).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds_pitch = []
    preds_yaw = []
    preds_roll = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            if model_type == "baseline":
                p = out.cpu().numpy().ravel()
                preds_pitch.extend(p)
            else:
                preds_pitch.extend(out[:, 0].cpu().numpy())
                preds_yaw.extend(out[:, 1].cpu().numpy())
                preds_roll.extend(out[:, 2].cpu().numpy())

    preds_pitch = np.array(preds_pitch)

    tgts_pitch = np.array([read_pose_mat(r["mat"])[0] for r in test_rows])
    tgts_yaw = np.array([read_pose_mat(r["mat"])[1] for r in test_rows])
    tgts_roll = np.array([read_pose_mat(r["mat"])[2] for r in test_rows])

    mask_screen = np.abs(tgts_yaw) < 45.0
    mask_extreme = ~mask_screen

    # Calculate metrics for Pitch
    diff_overall = preds_pitch - tgts_pitch
    diff_screen = preds_pitch[mask_screen] - tgts_pitch[mask_screen]
    diff_extreme = preds_pitch[mask_extreme] - tgts_pitch[mask_extreme]

    mse_overall = float(np.mean(diff_overall ** 2))
    rmse_overall = float(np.sqrt(mse_overall))
    mae_overall = float(np.mean(np.abs(diff_overall)))

    mse_screen = float(np.mean(diff_screen ** 2))
    rmse_screen = float(np.sqrt(mse_screen))
    mae_screen = float(np.mean(np.abs(diff_screen)))

    mse_extreme = float(np.mean(diff_extreme ** 2))
    rmse_extreme = float(np.sqrt(mse_extreme))
    mae_extreme = float(np.mean(np.abs(diff_extreme)))

    results = {
        "pitch": {
            "overall": {
                "n": len(test_rows),
                "mse": round(mse_overall, 4),
                "rmse": round(rmse_overall, 4),
                "mae": round(mae_overall, 4)
            },
            "screen_domain": {
                "n": int(np.sum(mask_screen)),
                "mse": round(mse_screen, 4),
                "rmse": round(rmse_screen, 4),
                "mae": round(mae_screen, 4)
            },
            "extreme_profile": {
                "n": int(np.sum(mask_extreme)),
                "mse": round(mse_extreme, 4),
                "rmse": round(rmse_extreme, 4),
                "mae": round(mae_extreme, 4)
            }
        }
    }

    if model_type == "tuned":
        preds_yaw = np.array(preds_yaw)
        preds_roll = np.array(preds_roll)
        results["yaw"] = {
            "overall_mse": round(float(np.mean((preds_yaw - tgts_yaw) ** 2)), 4),
            "overall_rmse": round(float(np.sqrt(np.mean((preds_yaw - tgts_yaw) ** 2))), 4),
            "overall_mae": round(float(np.mean(np.abs(preds_yaw - tgts_yaw))), 4),
        }
        results["roll"] = {
            "overall_mse": round(float(np.mean((preds_roll - tgts_roll) ** 2)), 4),
            "overall_rmse": round(float(np.sqrt(np.mean((preds_roll - tgts_roll) ** 2))), 4),
            "overall_mae": round(float(np.mean(np.abs(preds_roll - tgts_roll))), 4),
        }

    return results

def main():
    print("=" * 70, flush=True)
    print(" 📊 MENGHITUNG MSE (MEAN SQUARED ERROR) DAN RMSE PADA AFLW2000-3D", flush=True)
    print("=" * 70, flush=True)

    with open("ml/train/split_pitch.json") as f:
        test_rows = json.load(f)["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    models_to_eval = [
        ("Baseline (Single-Task)", "baseline", "ml/runs/pitch_20261118_181059/best.pt"),
        ("Tuned v1 (Multi-Task)", "tuned", "ml/runs/pitch_tuned_20261118_210456/best.pt"),
        ("Champion (Tuned + PitchWeight=2.0, Beta=1.0)", "tuned", "ml/runs/pitch_exp_pitchweight2_beta1/best.pt"),
    ]

    all_results = {}
    for name, mtype, ckpt in models_to_eval:
        print(f"\n--> Mengevaluasi {name}...", flush=True)
        res = evaluate_model_mse(mtype, ckpt, test_rows, device)
        all_results[name] = res

        p = res["pitch"]
        print(f"  [Screen Domain |yaw| < 45°]  MSE: {p['screen_domain']['mse']:8.4f}°² | RMSE: {p['screen_domain']['rmse']:6.4f}° | MAE: {p['screen_domain']['mae']:6.4f}°", flush=True)
        print(f"  [Extreme Profile |yaw|>=45°]  MSE: {p['extreme_profile']['mse']:8.4f}°² | RMSE: {p['extreme_profile']['rmse']:6.4f}° | MAE: {p['extreme_profile']['mae']:6.4f}°", flush=True)
        print(f"  [Overall AFLW2000-3D]         MSE: {p['overall']['mse']:8.4f}°² | RMSE: {p['overall']['rmse']:6.4f}° | MAE: {p['overall']['mae']:6.4f}°", flush=True)

    with open("ml/runs/mse_evaluation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ Evaluasi MSE selesai! Hasil disimpan ke ml/runs/mse_evaluation_summary.json", flush=True)

if __name__ == "__main__":
    main()
