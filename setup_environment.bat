@echo off
echo =====================================================================
echo  JagaMata ML - Setup Lingkungan PyTorch CUDA 12.8 + RTX 5050
echo  (Dukungan Native Blackwell sm_120 untuk NVIDIA GeForce RTX 50-Series)
echo =====================================================================

echo [1/3] Menginstall PyTorch dengan dukungan CUDA 12.8 (RTX 5050 Blackwell sm_120)...
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo [2/3] Menginstall dependensi ML pendukung...
python -m pip install timm opencv-python matplotlib scikit-learn scipy pandas tqdm pillow

echo.
echo [3/3] Memverifikasi deteksi GPU CUDA...
python -c "import torch; print('  PyTorch Version :', torch.__version__); print('  CUDA Available  :', torch.cuda.is_available()); print('  GPU Device Name :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Menjalankan diagnosa pre-flight JagaMata...
python ml\train\sanity_check.py

echo.
echo =====================================================================
echo  Setup selesai! Anda sekarang siap menjalankan training.
echo =====================================================================
pause
