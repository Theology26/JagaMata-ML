import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train_pitch import build_pitch_model, PoseDataset
from common import read_pose_mat

def main():
    with open("ml/train/split_pitch.json") as f:
        test_rows = json.load(f)["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pitch_model(pretrained=False).to(device)
    model.load_state_dict(torch.load("ml/runs/pitch_20261118_181059/best.pt", map_location=device))
    model.eval()

    loader = DataLoader(PoseDataset(test_rows, augment=False), batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            p = model(x.to(device)).cpu().numpy().ravel()
            preds.extend(p)
    preds = np.array(preds)

    tgts = np.array([read_pose_mat(r["mat"])[0] for r in test_rows])
    yaws = np.array([read_pose_mat(r["mat"])[1] for r in test_rows])

    mask_screen = np.abs(yaws) < 45.0
    mask_extreme = ~mask_screen

    overall_mae = float(np.mean(np.abs(preds - tgts)))
    screen_mae = float(np.mean(np.abs(preds[mask_screen] - tgts[mask_screen])))
    extreme_mae = float(np.mean(np.abs(preds[mask_extreme] - tgts[mask_extreme])))

    print(f"Total Test Samples   : {len(test_rows)}")
    print(f"Overall MAE          : {overall_mae:.4f}°")
    print(f"Screen (|yaw| < 45°) : N={int(np.sum(mask_screen))}, MAE={screen_mae:.4f}°")
    print(f"Extreme (|yaw|>= 45°): N={int(np.sum(mask_extreme))}, MAE={extreme_mae:.4f}°")

    # Update test_results.json for baseline
    res_path = "ml/runs/pitch_20261118_181059/test_results.json"
    with open(res_path, "r") as f:
        res = json.load(f)
    res["subgroup_screen_domain"] = {
        "condition": "|yaw| < 45° (Anak menatap layar gadget)",
        "n_samples": int(np.sum(mask_screen)),
        "test_mae_pitch": round(screen_mae, 4),
        "kpi_target": "<= 5.0°",
        "status": "LOLOS TARGET JagaMata ✅" if screen_mae <= 5.0 else "BELUM"
    }
    res["subgroup_extreme_profile"] = {
        "condition": "|yaw| >= 45° (Kepala menengok menyamping tajam)",
        "n_samples": int(np.sum(mask_extreme)),
        "test_mae_pitch": round(extreme_mae, 4),
        "diagnosa": "Oversmoothing / ambiguitas proyeksi 2D ke 3D"
    }
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[+] Updated {res_path} successfully!")

if __name__ == "__main__":
    main()
