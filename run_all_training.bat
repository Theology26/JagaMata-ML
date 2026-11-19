@echo off
echo =====================================================================
echo  JagaMata - Master Training Pipeline: Task 1 + Task 2 (Overnight Run)
echo  Hardware: NVIDIA GeForce RTX 5050 (AMP FP16) + Intel Core i7-13620H
echo =====================================================================

.venv\Scripts\python.exe -u ml\train\run_all_training.py
pause
