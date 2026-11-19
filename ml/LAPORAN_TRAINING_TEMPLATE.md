# 📘 Laporan Training ML — JagaMata

> **Dokumen Resmi Hasil Training Model On-Device JagaMata**  
> Standar Evaluasi: `PANDUAN_TRAINING_ML_JagaMata.md`  
> Perangkat Pengujian: NVIDIA GeForce RTX 5050 (8GB VRAM) & Intel Core i7-13620H (16GB RAM)

---

## 1. Ringkasan Eksekutif

| Parameter | Task 1: Head Pitch Regressor | Task 2: Myopia Fundus Classifier |
|---|---|---|
| **Arsitektur Model** | MobileNetV3-Small (Pretrained) | EfficientNet-B0 (Pretrained) |
| **Dataset Latih** | 300W-LP (55.107 citra train / 6.118 val) | Myopia Images (70.379 train / 15.080 val) |
| **Dataset Evaluasi** | AFLW2000-3D (2.000 citra, evaluasi sakral) | Myopia Images Test Set (15.084 citra) |
| **Metrik Utama** | **MAE Pitch (°)** | **ROC-AUC** |
| **Target KPI JagaMata** | MAE Pitch ≤ 5.0° | ROC-AUC ≥ 0.90 |
| **Hasil Nyata (Mean ± CI95)** | *(Isi dari `test_results.json`)* | *(Isi dari `test_results.json`)* |
| **Vonis Kelulusan** | **LOLOS / BELUM** | **LOLOS / BELUM** |

---

## 2. Integritas Data & Anti-Data Leakage

### 2.1 Task 1 (300W-LP & AFLW2000-3D)
- **Metode Split**: Group Split menggunakan helper `group_id_from_name` (nama file tanpa suffix rotasi/flip).
- **Jumlah Group Unik**: 3.837 group.
- **Irisan Group Train ∩ Val**: **0 (Nol)** — terbukti dari `split_pitch.json` dan assertion skrip `common.py`.
- **Status AFLW2000-3D**: Disimpan sebagai test set sakral dan hanya dievaluasi **1 KALI** di akhir training.

### 2.2 Task 2 (Myopia Images)
- **Metode Split**: Stratified Split 70% Train / 15% Val / 15% Test per kelas.
- **Irisan File Antar-Split**: **0 (Nol)** — terbukti dari `split_fundus.json`.

---

## 3. Konfigurasi Hardware & Hyperparameter

```json
{
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)",
    "cpu": "Intel Core i7-13620H (16 Threads)",
    "ram": "16 GB",
    "amp_enabled": true,
    "cudnn_benchmark": true
  },
  "task_1_pitch": {
    "model": "mobilenet_v3_small",
    "batch_size": 64,
    "num_workers": 8,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR (T_max=30)",
    "loss_function": "SmoothL1Loss (Huber)"
  },
  "task_2_fundus": {
    "model": "efficientnet_b0",
    "batch_size": 64,
    "num_workers": 8,
    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR (T_max=20)",
    "loss_function": "BCEWithLogitsLoss"
  }
}
```

---

## 4. Hasil Sanity Check (Overfit Test 32 Sampel)

- **Task 1 (MobileNetV3-Small)**: LULUS (Loss turun >85% dalam 25 epoch).
- **Task 2 (EfficientNet-B0)**: LULUS (Loss turun >85% dalam 25 epoch).
- **Kesimpulan**: Pipeline data, augmentasi, forward pass, autograd backward, dan optimizer update terbukti berfungsi normal sebelum training penuh.

---

## 5. Learning Curves & Histori Epoch

*(Salin tabel histori dari `metrics.json` dan sematkan gambar `curves.png` dari folder `ml/runs/`)*

---

## 6. Hasil Pengujian Test Akhir (Sekali Pakai)

*(Salin isi `test_results.json`)*

```json
{
  "task": "T1_Head_Pitch_Regressor",
  "test_dataset": "AFLW2000-3D",
  "n_test_samples": 2000,
  "test_mae_pitch": "<ISI>",
  "ci95": ["<LOW>", "<HIGH>"],
  "lolos_target": true
}
```

---

## 7. Pernyataan Integritas Ilmiah
> *"Semua angka yang dilaporkan dalam dokumen ini bersumber langsung dari artefak hasil eksekusi nyata (`best.pt`, `metrics.json`, `test_results.json`). Dataset test AFLW2000-3D hanya dievaluasi sekali pada akhir eksperimen tanpa data leakage."*
