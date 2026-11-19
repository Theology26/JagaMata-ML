# 📘 PANDUAN TRAINING ML — JAGAMATA
### Panduan langkah-demi-langkah untuk manusia: melatih & mengevaluasi model dari 6 dataset JagaMata

> **Untuk siapa:** rekan tim yang mengerjakan bagian ML project JagaMata
> **Level:** menengah (pernah pakai Python; belum tentu pernah training PyTorch penuh)
> **Target akhir:** model **head pitch MAE ≤ 5°** + laporan training jujur
> **Estimasi waktu total:** ±10–20 jam (RTX 3070) — wajib cuma **T1 (±4–8 jam, semalaman)**

---

## 0. BACA DULU — Apa yang DI-training dan TIDAK

| # | Dataset | Peran | Di-training? |
|---|---|---|---|
| 1 | `300W-LP` (122rb img) | Latih **T1 Head Pitch** | ✅ **WAJIB — ini inti** |
| 2 | `myopia-images` (100rb img) | Latih **T2 Myopia Classifier** | ⚠️ Opsional |
| 3 | `aflw2000-3d` (2.000 img) | **TEST FINAL** T1 | ❌ **DILARANG ditrain** (bocor!) |
| 4 | `wider-face` (16.106 img) | Evaluasi deteksi wajah | ❌ Inference saja |
| 5 | `utkface` (23.708 img) | Evaluasi wajah anak | ❌ Inference saja |
| 6 | `myopia-study/myopia.csv` | Konteks riset (tabel) | ❌ Tidak ada gambar |

**Aturan emas:**
1. **Test set sakral** — AFLW2000 hanya dipegang SEKALI di akhir. Sebelum itu, semua keputusan pakai **validation**.
2. **Angka harus nyata** — semua metrik dari file `metrics.json` hasil run, bukan dari ingatan.
3. **Split dulu, augmentasi kemudian.**

---

## 1. Prasyarat

| Kebutuhan | Minimum (rekomendasi) |
|---|---|
| GPU | **NVIDIA ≥6GB VRAM** (RTX 3070 kamu: 8GB ✅) — CPU-only = 5–10× lebih lambat |
| RAM | 16 GB (32 GB enak buat dataloader) |
| Disk kosong | ±25 GB (13,5 GB data + 5 GB env + hasil run) |
| Python | 3.10–3.12 |
| Akun | **Kaggle** (untuk unduh dataset) |

### 1.1 Install environment (sekali jalan)

```bash
# 1. Buat virtual env (di folder project)
cd "D:\Semester 5\Mobile Application\JagaMataApp"
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. PyTorch + CUDA (cek dulu versi driver: nvidia-smi → "CUDA Version: 12.x")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Deps lain
pip install timm opencv-python matplotlib scikit-learn scipy pandas tqdm pillow

# 4. VERIFIKASI WAJIB — harus keluar `cuda True`
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ Jika `cuda False`: cek driver NVIDIA via `nvidia-smi`, install ulang driver, ulangi step 2.

### 1.2 Struktur folder (pastikan seperti ini)

```
JagaMataApp/
├── ml/
│   ├── datasets/                  # sudah ada (unduh dari Kaggle, link §2)
│   │   ├── 300W-LP/…, aflw2000-3d/…, wider-face/, utkface/,
│   │   ├── myopia-images/, myopia-study/, DATASETS.md
│   ├── train/
│   │   ├── common.py              # split, label loader, utils (buat di §4)
│   │   ├── train_pitch.py         # T1 (buat di §5)
│   │   └── train_fundus.py        # T2 (opsional, pola sama §7)
│   ├── runs/                      # hasil training (otomatis terbuat)
│   └── LAPORAN_TRAINING.md        # laporan akhir (§10)
```

---

## 2. Unduh Dataset (kalau belum ada)

Link & verifikasi jumlah ada di **`ml/datasets/DATASETS.md`**. Versi CLI:

```bash
kaggle datasets download -d kameo4189/aflw2000-300wlp -p ml/datasets/aflw2000-300wlp --unzip
kaggle datasets download -d mohamedadlyi/aflw2000-3d   -p ml/datasets/aflw2000-3d   --unzip
kaggle datasets download -d mksaad/wider-face-a-face-detection-benchmark -p ml/datasets/wider-face --unzip
kaggle datasets download -d jangedoo/utkface-new        -p ml/datasets/utkface       --unzip
kaggle datasets download -d kellysanderson/myopia-images -p ml/datasets/myopia-images --unzip
kaggle datasets download -d mscgeorges/myopia-study     -p ml/datasets/myopia-study  --unzip
```

**Verifikasi sebelum training** (hitung = bukti unduhan utuh):
```bash
# Harus: 122450 | 2000 | 16106 | 23708 | 100543
find ml/datasets/aflw2000-300wlp/300W-LP -name "*.jpg" | wc -l
find ml/datasets/aflw2000-3d/AFLW2000    -name "*.jpg" | wc -l
find ml/datasets/wider-face               -name "*.jpg" | wc -l
find ml/datasets/utkface/UTKFace          -name "*.jpg" | wc -l
find ml/datasets/myopia-images            -name "*.jpg" | wc -l
```

---

## 3. Fakta Data yang WAJIB kamu tahu (anti-bocor)

### 3.1 300W-LP — data sintetis berkelompok ⚠️
```
Path : ml/datasets/aflw2000-300wlp/300W-LP/300W_LP/{AFW, HELEN, IBUG, LFPW, *_Flip}/
Contoh: AFW_1051618982_1_0.jpg
        AFW_1051618982_1_1.jpg   ← ROTASI dari foto yang SAMA
        AFW_1051618982_1_10.jpg
```
- **Satu foto asli → puluhan salinan rotasi.** Kalau salinan kembar terpecah train/val → **metrik BOHONG**.
- **Group ID = nama file tanpa elemen terakhir** → `AFW_1051618982_1`
- Folder `*_Flip` = cerminan → nama file **identik** dengan aslinya → otomatis ikut group yang sama ✅

### 3.2 AFLW2000-3D — test final
```
Path : ml/datasets/aflw2000-3d/AFLW2000/
Contoh: image00002.jpg + image00002.mat
Label: .mat berisi Pose_Pitch, Pose_Yaw, Pose_Roll (RADIAN) + landmark 3D
```

### 3.3 UTKFace — 3 varian duplikat
Pakai **hanya** `utkface/UTKFace/` (yang lain `crop_part1/`, `utkface_aligned_cropped/` = salinan). Label di nama file: `age_gender_race_tanggal.jpg` → **anak = age < 13** (3.413 foto).

### 3.4 Myopia fundus
```
ml/datasets/myopia-images/Images/Myopia_images/  → kelas 1 (47.025)
ml/datasets/myopia-images/Images/Normal_images/  → kelas 0 (53.518)
```

---

## 4. TAHAP 0 — Persiapan Label & Split (30 menit)

### 4.1 Ekstrak label pitch/yaw/roll → CSV

Buat `ml/train/common.py`:

```python
"""Label extractor + group split untuk JagaMata ML."""
import os, re, random, json
from pathlib import Path
import numpy as np
import scipy.io as sio

DS = Path("ml/datasets")
PITCH_TEST = DS / "aflw2000-3d/AFLW2000"
LP_ROOT    = DS / "aflw2000-300wlp/300W-LP/300W_LP"
FLIP_DIRS  = {"AFW_Flip", "HELEN_Flip", "IBUG_Flip", "LFPW_Flip"}

def read_pose_mat(mat_path):
    """→ (pitch_deg, yaw_deg, roll_deg) — konversi radian → derajat."""
    m = sio.loadmat(str(mat_path))
    # Tunjukkan key sekali kalau format beda:
    # print([k for k in m if not k.startswith('__')])
    def get(*names):
        for n in names:
            if n in m: return float(np.asarray(m[n]).squeeze())
        raise KeyError(f"pose key tidak ada; keys = {[k for k in m if not k.startswith('__')]}")
    pitch = np.degrees(get("Pose_Pitch", "pitch"))
    yaw   = np.degrees(get("Pose_Yaw",   "yaw"))
    roll  = np.degrees(get("Pose_Roll",  "roll"))
    return pitch, yaw, roll

def group_id_from_name(fname: str) -> str:
    """AFW_1051618982_1_0.jpg → AFW_1051618982_1  (buang suffix rotasi)."""
    stem = Path(fname).stem
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem

def build_lp_index():
    """Scan 300W-LP → list of dict (jpg, mat, group). Skip kalau mat hilang."""
    rows = []
    for ds_name in ["AFW", "HELEN", "IBUG", "LFPW"]:        # asli saja (Flip = duplikat cermin,
        d = LP_ROOT / ds_name                                #         cukup pakai asli + augmentasi flip
        if not d.exists(): continue
        for jpg in sorted(d.glob("*.jpg")):
            mat = jpg.with_suffix(".mat")
            if mat.exists():
                rows.append({"jpg": str(jpg), "mat": str(mat),
                             "group": group_id_from_name(jpg.name)})
    return rows

def build_test_index():
    rows = []
    for jpg in sorted(PITCH_TEST.glob("*.jpg")):
        mat = jpg.with_suffix(".mat")
        if mat.exists():
            rows.append({"jpg": str(jpg), "mat": str(mat), "group": jpg.stem})
    return rows

def group_split(rows, val_frac=0.10, seed=42):
    """Split BERBASIS GROUP — tidak ada foto asli yang terbelah."""
    groups = sorted({r["group"] for r in rows})
    random.Random(seed).shuffle(groups)
    n_val = int(len(groups) * val_frac)
    val_groups = set(groups[:n_val])
    train = [r for r in rows if r["group"] not in val_groups]
    val   = [r for r in rows if r["group"] in val_groups]
    return train, val

if __name__ == "__main__":
    print("Scan 300W-LP ...")
    lp = build_lp_index()
    print(f"  {len(lp)} pasang jpg+mat, {len({r['group'] for r in lp})} group unik")
    tr, va = group_split(lp)
    # VERIFIKASI anti-leakage: irisan group harus KOSONG
    assert not ({r['group'] for r in tr} & {r['group'] for r in va}), "LEAKAGE!"
    print(f"  train={len(tr)}  val={len(va)}  irisan group=0 ✅")
    print("Scan AFLW2000 (test, sekali pakai)...")
    te = build_test_index()
    print(f"  test={len(te)}")
    json.dump({"train": tr, "val": va, "test": te},
              open("ml/train/split.json", "w"), indent=1)
    print("→ ml/train/split.json tersimpan")
```

Jalankan:
```bash
python -m ml.train.common     # harus: irisan group=0 ✅
```

> **Kalau ada `KeyError` saat baca `.mat`:** cetak dulu key-nya (baris `print(...)` sudah disiapkan) — format mirror bisa beda (`Pose_Pitch` vs `pose_pitch` vs tabel `pose`). Sesuaikan fungsi `get()` dengan key yang muncul.

### 4.2 Sanity check WAJIB sebelum training penuh (10 menit)

```python
# Uji cepat: model harus BISA overfit 32 sampel. Gagal = pipeline rusak.
# (training 32 sampel, 20 epoch, loss harus turun mendekati 0)
```
- **Lulus:** loss epoch akhir < 10% loss awal → pipeline sehat, lanjut.
- **Gagal:** cek label terbalik, LR terlalu kecil/besar, normalisasi input, gradient mengalir (`loss.backward()` dipanggil).

---

## 5. T1 — TRAINING HEAD PITCH (inti, ±4–8 jam)

Buat `ml/train/train_pitch.py`:

```python
"""T1: Head pitch regressor — 300W-LP train, AFLW2000 test (sekali pakai)."""
import json, time, math, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from common import read_pose_mat

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

RUN = Path(f"ml/runs/pitch_{time.strftime('%Y%m%d_%H%M')}"); RUN.mkdir(parents=True, exist_ok=True)
TARGET = "pitch"            # label utama JagaMata (kemudian boleh tambah yaw/roll)

# ---------- data ----------
class PoseDataset(Dataset):
    def __init__(self, rows, augment=False, target=TARGET):
        self.rows, self.aug, self.target = rows, augment, target
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(0.5),   # aman: pitch tidak berubah saat flip
            transforms.ColorJitter(0.3, 0.3, 0.3),  # brightness/contrast/saturation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]) if augment else transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = Image.open(r["jpg"]).convert("RGB")
        p, y, ro = read_pose_mat(r["mat"])
        label = {"pitch": p, "yaw": y, "roll": ro}[self.target]
        return self.tf(x), torch.tensor([label], dtype=torch.float32)

# ---------- model ----------
def make_model():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)   # output: 1 (pitch)
    return m

# ---------- metrik ----------
def mae(pred, tgt): return float((pred - tgt).abs().mean())

# ---------- training ----------
def main():
    split = json.load(open("ml/train/split.json"))
    train_rows, val_rows, test_rows = split["train"], split["val"], split["test"]

    tr_dl = DataLoader(PoseDataset(train_rows, augment=True), batch_size=64, shuffle=True,
                       num_workers=8, pin_memory=True)
    va_dl = DataLoader(PoseDataset(val_rows), batch_size=128, num_workers=4, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    lossf = nn.SmoothL1Loss()                       # Huber → tahan label noise

    best_val, patience, bad = math.inf, 15, 0
    history = []
    for epoch in range(1, 31):
        model.train(); t0 = time.time(); tr_loss = 0; n = 0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = model(x); loss = lossf(out, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(x); n += len(x)
        tr_loss /= n

        model.eval(); v_pred, v_tgt = [], []
        with torch.no_grad():
            for x, y in va_dl:
                x = x.to(dev); v_pred.append(model(x).cpu()); v_tgt.append(y)
        v_pred = torch.cat(v_pred); v_tgt = torch.cat(v_tgt)
        v_mae = mae(v_pred, v_tgt)
        sched.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_mae": round(v_mae, 4), "lr": opt.param_groups[0]["lr"],
                        "sec": round(time.time() - t0, 1)})
        print(f"epoch {epoch:02d} | loss {tr_loss:.4f} | val MAE {v_mae:.2f}° | "
              f"{history[-1]['sec']}s | best {best_val:.2f}°")

        # early stopping pada val MAE terbaik
        if v_mae < best_val - 0.01:
            best_val, bad = v_mae, 0
            torch.save(model.state_dict(), RUN / "best.pt")
            json.dump(history, open(RUN / "metrics.json", "w"), indent=1)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop di epoch {epoch} (best val MAE {best_val:.2f}°)")
                break

    # ---------- EVALUASI TEST: SEKALI SAJA ----------
    print("\n=== TEST FINAL AFLW2000 (sekali pakai) ===")
    model.load_state_dict(torch.load(RUN / "best.pt"))
    model.eval()
    te_dl = DataLoader(PoseDataset(test_rows), batch_size=128, num_workers=4)
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in te_dl:
            preds.append(model(x.to(dev)).cpu()); tgts.append(y)
    preds = torch.cat(preds).numpy().ravel(); tgts = torch.cat(tgts).numpy().ravel()

    # bootstrap 95% CI (1.000 resample)
    rng = np.random.default_rng(SEED)
    boots = [np.mean(np.abs(preds[rng.integers(0, len(preds), len(preds))] - tgts[
              rng.integers(0, len(tgts), len(tgts))])) for _ in range(1000)]
    ci = np.percentile(boots, [2.5, 97.5])

    result = {"test_mae_pitch": float(np.mean(np.abs(preds - tgts))),
              "ci95": [float(ci[0]), float(ci[1])],
              "val_mae_best": float(best_val),
              "n_test": len(test_rows),
              "target": "MAE pitch <= 5.0 degree",
              "lolos": bool(np.mean(np.abs(preds - tgts)) <= 5.0)}
    json.dump(result, open(RUN / "test_results.json", "w"), indent=1)
    print(json.dumps(result, indent=1))
    print("\nHistori epoch →", RUN / "metrics.json")
    # plot learning curves:
    # import matplotlib.pyplot as plt; plt.plot([h['train_loss'] for h in history], label='train_loss')
    # plt.plot([h['val_mae'] for h in history], label='val_mae'); plt.legend(); plt.savefig(RUN/'curves.png')

if __name__ == "__main__":
    main()
```

Jalankan:
```bash
cd "D:\Semester 5\Mobile Application\JagaMataApp"
python ml/train/train_pitch.py
```

**Yang kamu lihat & artinya:**

| Pola di log | Arti | Tindakan |
|---|---|---|
| `train_loss` turun, `val MAE` turun beriringan | ✅ sehat | lanjut sampai early stop |
| `train_loss` turun, `val MAE` naik terus | 🔴 **OVERFIT** | lihat §8 baris 1 |
| Keduanya tinggi & datar dari awal | 🔵 **UNDERFIT / pipeline rusak** | sanity check §4.2 dulu |
| `val MAE < 5.0°` di `test_results.json` | 🎯 **TARGET TERCAPAI** | tulis laporan §10 |

**Target:** `test_mae_pitch ≤ 5.0°` (KPI K-pitch JagaMata).

---

## 6. EKSPERIMEN YANG BOLEH (tanpa menyentuh test)

Satu run = satu hipotesis. Catat semua di tabel:

| # | Perubahan | val MAE | Kesimpulan |
|---|---|---|---|
| 1 | Baseline (MobileNetV3, lr 1e-3) | …° | dasar |
| 2 | + weight decay 1e-3, dropout head 0,3 | …° | |
| 3 | ResNet-18 (kapasitas naik) | …° | hanya jika #2 masih underfit |
| 4 | lr 3e-4 + warmup 3 epoch | …° | |

**Aturan:** paling banyak 3–5 run sebelum test final. Pilih **val MAE terbaik**, lalu test **sekali**.

---

## 7. T2 — Myopia Fundus (OPSIONAL, ±4–8 jam)

Pola sama, bedanya:
- Split: **stratified 70/15/15 per kelas** (bukan group — per kelas), seed tetap.
- Model: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)`
- Loss: `BCEWithLogitsLoss`; metrik: **AUC** (utama), F1, sensitivity, specificity.
- Target: **AUC ≥ 0,90**.
- ⚠️ Augmentasi **setelah** split; jangan ada file kembar beda split.
- Laporkan subgroup: per sumber folder (myopia vs normal) — kalau metrik per kelas timpang >10 poin → cek keseimbangan & kualitas label.

## 8. EVALUASI (inference saja, cepat)

**WIDER FACE — recall deteksi wajah (K1):** jalankan detektor (MediaPipe/ML Kit atau detektor kamu) di `wider-face/WIDER_val/images` (3.226 img), bandingkan dengan bbox di `wider_face_split/wider_face_val_bbx_gt.txt`. Hitung recall@IoU 0,5 → target **≥90%**.

**UTKFace anak:** filter `age<13` (3.413) → jalankan pipeline deteksi → bandingkan performa vs subset dewasa (selisih <15 poin = K1 lintas usia lolos).

---

## 9. DIAGNOSIS — Overfit / Underfit / Bias

### 🔴 Overfitting (val naik, train turun)
Urutan perbaikan: **1)** early stopping (sudah ada) **2)** augmentasi lebih kaya **3)** weight decay 1e-4→1e-3 + dropout 0,2–0,4 **4)** model lebih kecil **5)** data lebih banyak (122rb masih banyak yang belum dipakai).

### 🔵 Underfitting (dua-duanya buruk)
Urutan: **1)** sanity check 32 sampel **2)** kalahkan baseline mean **3)** model lebih besar / epoch lebih banyak **4)** LR naik **5)** audit 50 label manual.

### 🟡 Bias (subgroup timpang >10 poin)
Wajib laporan tabel metrik per: **age bucket** (UTKFace: 0–5/6–12/13–30/31+), **gender**, **race** (audit saja — **bukan fitur model**), **kelas fundus**.
- Timpang → **rebalance sampling** → augmentasi representatif → ulangi → laporkan angka sebelum & sesudah.
- Ingat: UTKFace = selfie ter-crop; kamera anak asli beda domain → **jangan klaim generalisasi** tanpa uji di data D1 (180 foto penggaris, dikumpulkan manual).

---

## 10. TROUBLESHOOTING CEPAT

| Gejala | Penyebab | Solusi |
|---|---|---|
| `CUDA out of memory` | Batch/resolusi kebesaran | batch 64→32, atau `torch.cuda.empty_cache()`, atau `torch.backends.cudnn.benchmark=False` |
| `cuda False` | Driver/torch CUDA tak cocok | `nvidia-smi` cek versi → install torch `cuXXX` yang cocok |
| Loss `nan` | LR kebesaran / data rusak | lr ÷10; cek 10 gambar manual; `torch.nan_to_num` bukan solusi — cari akarnya |
| Training lambat (~0 iter/s) | `num_workers=0` / disk lambat | `num_workers=8, pin_memory=True`; pindah SSD |
| `KeyError` baca `.mat` | Format beda | cetak `m.keys()` → sesuaikan `get()` (§4.1) |
| val MAE aneh kecil (mis. 0,5°) | **Leakage** | cek irisan group = 0; pastikan flip & rotasi kembar tak terbelah |
| Overfit epoch 3 | Kapasitas kebesaran / data kekecilan | model lebih kecil + augmentasi + pakai lebih banyak group |

---

## 11. CEKLIST SELESAI (Definition of Done)

```
ENV
☐ torch.cuda = True · deps terpasang · verifikasi keluar `cuda True`
DATA
☐ split.json tersimpan · irisan group = 0 (terbukti log) · test belum disentuh
SANITY
☐ Overfit 32 sampel lulus (pipeline sehat)
TRAINING T1
☐ ≥1 run penuh dengan early stopping · best.pt + metrics.json + curves.png tersimpan
☐ Pilih model dari VAL terbaik (bukan test)
EVALUASI
☐ Test AFLW2000 dijalankan SEKALI → test_results.json berisi MAE + 95% CI
☐ MAE pitch ≤5° ATAU rencana perbaikan tertulis
FAIRNESS & ERROR
☐ Tabel subgroup (age/gender/race) untuk evaluasi UTKFace
☐ 20 sampel error terburuk dianalisis
LAPORAN
☐ ml/LAPORAN_TRAINING.md lengkap (format di bawah), semua angka dari file run
```

**Format laporan (salin):**
```markdown
# Laporan Training — <task> — <tanggal>
## 1. Ringkasan        : model, data, MAE/AUC ± CI, veredik
## 2. Split & Leakage  : bukti irisan group=0, seed
## 3. Konfigurasi      : tabel hyperparameter
## 4. Learning Curves  : analisis + kurva.png
## 5. Hasil Test       : metrik ± 95% CI (sekali pakai)
## 6. Subgroup/Fairness: tabel per subgroup + gap
## 7. Error Analysis   : 20 sampel terburuk + pola
## 8. Diagnosa         : overfit/underfit/bias (atau "tidak ada") + bukti
## 9. Langkah Berikut  : perbaikan / rekomendasi rilis
## 10. Integritas      : "semua angka dari run nyata; test disentuh sekali"
```

---

## 12. JADWAL YANG DISARANKAN

```
MALAM 0 (30 mnt) : env + download + §4 (split & sanity check)
MALAM 1 (4–8 jam): T1 training jalan sendiri → tidur
PAGI 1 (1 jam)   : baca hasil → kalau MAE >5° → 1 run perbaikan (§6)
MALAM 2 (opsi)   : T2 fundus / evaluasi WIDER + UTKFace
HARI 3 (1 jam)   : LAPORAN_TRAINING.md → selesai ✅
```

---

*Panduan Training JagaMata v1.0 · Dataset: katalog `ml/datasets/DATASETS.md` · Kontrak agent: `PROMPT_Training_ML_JagaMata.md`*
