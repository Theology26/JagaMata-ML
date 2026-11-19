@echo off
echo =====================================================================
echo  JagaMata - Task 2: Training Myopia Fundus Classifier (EfficientNet-B0)
echo  Hardware: NVIDIA RTX 5050 (AMP) + Intel Core i7-13620H
echo  Dataset : Myopia Images (70%% Train / 15%% Val / 15%% Test)
echo =====================================================================

.venv\Scripts\python.exe -u ml\train\train_fundus.py --batch-size 64 --epochs 20 --num-workers 8 --amp
pause
