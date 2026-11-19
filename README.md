# JagaMata ML: On-Device Computer Vision & Pose Regression

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20%2B-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Android Stack](https://img.shields.io/badge/Android-100%25%20Kotlin%20%2B%20Jetpack%20Compose-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Repository ini memuat pipeline machine learning, arsitektur model, skrip pelatihan, evaluasi empiris, dan modul ekspor ONNX untuk ekosistem **JagaMata** — sistem pencegahan miopia dini dan koreksi ergonomi postur layar gawai pada anak berbasis On-Device AI.

---

## 📌 Ringkasan Arsitektur & Model Champion

| Komponen | Task 1: Head Pitch Regressor (Champion) | Task 2: Myopia Screening (Research) |
|---|---|---|
| **Backbone** | MobileNetV3-Small (Pretrained ImageNet-1K) | EfficientNet-B0 (Pretrained ImageNet-1K) |
| **Tipe Output** | Multi-Task Joint 3D Pose `[pitch, yaw, roll]` | Probabilitas Biner (Normal vs Myopia) |
| **Fungsi Loss** | SmoothL1Loss (Huber $\beta=1.0$), Pitch-Weight 2.0x | Binary Cross-Entropy with Logits |
| **Ukuran Model ONNX INT8** | **1.69 MB** (Dynamic Quantized) | 16.01 MB (FP32) |
| **Estimasi Latensi Mobile** | ~12 ms / frame (NPU / CPU Snapdragon) | ~45 ms / inference |
| **Dataset Uji Sakral** | AFLW2000-3D ($N = 2.000$ citra wajah riil) | Myopia Images Test Split ($N = 15.084$ fundus) |
| **Performa Utama (Screen Domain)** | **MAE = 4.3692°** (Target KPI: $\le 5.0^\circ$ ✅) | **ROC-AUC = 1.0000**, F1 = 1.0000 |
| **MSE / RMSE (Screen Domain)** | **MSE = 42.0147 °²**, **RMSE = 6.4819°** | Brier Score $\approx 0.0000$ |
| **Target Integrasi Mobile** | **100% Native Kotlin + CameraX + MediaPipe** | Modul Tele-Skrining Optik (Shadow Running) |

---

## 📂 Struktur Direktori Proyek

```text
JagaMata-ML/
├── .gitignore                          # Menjaga repo tetap ringan (dataset & venv diabaikan)
├── Blueprint_Perusahaan_JagaMata_PerDivisi.md # Arsitektur sistem 100% Kotlin + Compose
├── PANDUAN_TRAINING_ML_JagaMata.md     # Spesifikasi ketat machine learning JagaMata
├── README.md                           # Dokumentasi utama proyek
├── run_all_training.bat                # Eksekutor batch training Windows
├── run_train_fundus.bat                # Eksekutor training Task 2
├── run_train_pitch.bat                 # Eksekutor training Task 1
├── setup_environment.bat               # Inisialisasi dependensi & PyTorch CUDA
└── ml/
    ├── LAPORAN_TRAINING.md             # Laporan saintifik komprehensif (Audit KPI & MSE)
    ├── LAPORAN_TRAINING_2.md           # Sinkronisasi laporan iterasi kedua
    ├── export/
    │   └── export_models.py            # Skrip ekspor PyTorch -> ONNX FP32 & INT8 Quantized
    ├── models/                         # Model siap pakai untuk Android (ONNX)
    │   ├── ANDROID_INTEGRATION_GUIDE.md# Panduan integrasi Kotlin & CameraX
    │   ├── export_summary.json         # Metadata dimensi input/output model
    │   ├── jagamata_pitch_tuned_int8.onnx   # Model utama deployment Android (1.69 MB)
    │   ├── jagamata_pitch_tuned.onnx        # Model FP32 pitch (6.09 MB)
    │   └── jagamata_fundus_classifier.onnx  # Model FP32 fundus (16.01 MB)
    ├── runs/                           # Checkpoints, kurva loss, dan log evaluasi JSON
    │   ├── hyperparameter_tuning_summary.json
    │   ├── mse_evaluation_summary.json
    │   ├── top20_worst_errors.json
    │   ├── pitch_exp_pitchweight2_beta1/ # Run Champion Model Task 1
    │   └── fundus_20261118_200337/       # Run Model Task 2
    └── train/                          # Skrip source code pelatihan & evaluasi
        ├── common.py                   # Pembaca .mat, rotasi biner, stratified split
        ├── compute_mse.py              # Skrip komputasi metrik MSE & RMSE
        ├── error_analysis_top20.py     # Audit kegagalan 20 sampel terburuk AFLW2000
        ├── train_pitch.py              # Baseline MobileNetV3 (Single-Task)
        ├── train_pitch_tuned.py        # Tuned Multi-Task 3D Pose Regressor
        ├── train_fundus.py             # EfficientNet-B0 Fundus Classifier
        ├── split_pitch.json            # Index split 300W-LP anti-leakage
        └── split_fundus.json           # Index split Myopia images stratified 70/15/15
```

---

## 🔬 Metodologi & Pencegahan Data Leakage

1. **Anti-Leakage Group Split (Task 1):**  
   Dataset `300W-LP` memuat ribuan citra sintesis dari orang yang sama. Split dilakukan berbasis **ID Subjek/Group** (`irisan subjek Train, Val, Test = 0`). Data uji sakral menggunakan **AFLW2000-3D** ($N = 2.000$) yang sepenuhnya independen dari dataset latih.
2. **Evaluasi Subgroup Apples-to-Apples:**  
   Evaluasi AFLW2000-3D dipilah secara objektif membaca ground-truth matriks pose:
   - **Screen Domain ($|\text{yaw}| < 45^\circ$, $N = 1.508$):** Rentang fungsional anak menatap layar ponsel. Champion model mencapai **MAE 4.37°** dan **MSE 42.01 °²** (Lolos KPI $\le 5.0^\circ$).
   - **Extreme Profile Domain ($|\text{yaw}| \ge 45^\circ$, $N = 492$):** Rentang kepala menoleh tajam ke samping. Model multi-task terbukti meredam *oversmoothing* dan memangkas error dari 25.10° ke 23.65°.
3. **Audit Kritis Shortcut Learning (Task 2):**  
   Meskipun EfficientNet-B0 mencapai AUC 1.0000, analisis mendalam mengungkap adanya disparitas visual non-patologis antar-folder (ketebalan batas mask lingkaran hitam dan kompresi JPEG). Status model ini diposisikan pada **mode riset / shadow-running** sebelum diuji pada dataset multi-senter eksternal.

---

## 🚀 Panduan Menjalankan Proyek

### 1. Prasyarat Lingkungan
- OS: Windows 10/11 atau Linux (Ubuntu 22.04+)
- Python 3.10 - 3.12
- GPU NVIDIA dengan dukungan CUDA (opsional, CPU fallback didukung)

### 2. Instalasi Dependensi
```bash
# Clone repository (sangat cepat karena tanpa raw dataset)
git clone https://github.com/Theology26/JagaMata-ML.git
cd JagaMata-ML

# Buat virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch dengan CUDA 12.4 / 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install pustaka penunjang
pip install numpy pillow scipy matplotlib onnx onnxruntime onnxscript
```

### 3. Persiapan Dataset (Opsional untuk Training Ulang)
Dataset mentah sengaja **tidak dimasukkan ke git** agar repositori tetap ringan dan proses cloning berlangsung instan. Jika ingin melatih ulang:
- **Task 1 (Head Pose):** Unduh `300W-LP` dan `AFLW2000-3D`, letakkan di folder `AFLW2000_300W-LP/`.
- **Task 2 (Myopia):** Unduh dataset dari Kaggle `kellysanderson/myopia-images`, letakkan citra di folder `Myopia images/Images/`.

### 4. Eksekusi Pelatihan & Evaluasi
```bash
# Menjalankan Sanity Check (32 sampel)
python ml/train/sanity_check.py

# Pelatihan Model Champion Task 1 (Multi-Task 3D Pose Regressor)
python ml/train/train_pitch_tuned.py --epochs 30 --batch-size 64 --amp

# Komputasi Metrik MSE & RMSE pada AFLW2000-3D
python ml/train/compute_mse.py

# Audit Top 20 Sampel Error Terburuk
python ml/train/error_analysis_top20.py
```

### 5. Ekspor Model ke ONNX & Quantization INT8
Model yang siap dideploy sudah tersedia di folder `ml/models/`. Jika ingin mengekspor ulang dari checkpoint `.pt`:
```bash
python ml/export/export_models.py
```

---

## 📱 Integrasi ke Android (100% Native Kotlin)

Model inferensi utama adalah **`ml/models/jagamata_pitch_tuned_int8.onnx`** (hanya berukuran **1.69 MB**).

### Arsitektur Runtime On-Device:
1. **Google MediaPipe FaceLandmarker:** Mendeteksi wajah secara real-time via CameraX dan mengekstrak ROI wajah bounding box.
2. **Crop & Normalisasi:** Citra wajah di-crop ke resolusi $224 \times 224 \times 3$, dinormalisasi ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
3. **ONNX Runtime Android:** Menjalankan inferensi INT8 quantized tensor dengan latensi ~12 ms.
4. **Logic Trigger Koreksi:** Jika sudut `Pitch < -20°` (anak menunduk tajam ke layar) selama lebih dari 5 detik berturut-turut, sistem Jetpack Compose memunculkan overlay peringatan ergonomis.

Panduan lengkap implementasi Kotlin, CameraX ImageAnalysis, dan kode siap pakai dapat dibaca di:
👉 [**ml/models/ANDROID_INTEGRATION_GUIDE.md**](ml/models/ANDROID_INTEGRATION_GUIDE.md)

---

## 📑 Laporan Lengkap & Hasil Eksperimen
Detail komparasi ilmiah, visualisasi kurva loss, analisis failure modes, dan evaluasi bootstrap interval konfidensi dapat dipelajari di:
👉 [**ml/LAPORAN_TRAINING.md**](ml/LAPORAN_TRAINING.md)

---

## 📄 Lisensi & Kontributor
Proyek ini dikembangkan di bawah lisensi MIT untuk pengembangan riset dan inovasi teknologi kesehatan mata anak di Indonesia.
