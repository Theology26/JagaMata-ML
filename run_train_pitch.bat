@echo off
echo =====================================================================
echo  JagaMata - Task 1: Training Head Pitch Regressor (MobileNetV3-Small)
echo  Hardware: NVIDIA RTX 5050 (AMP) + Intel Core i7-13620H
echo  Dataset : 300W-LP (Train/Val) | AFLW2000-3D (Test Final Sakral)
echo =====================================================================

.venv\Scripts\python.exe -u ml\train\train_pitch.py --batch-size 64 --epochs 30 --num-workers 8 --amp
pause
