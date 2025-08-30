@echo off
echo ============================================================
echo HRM Laptop Training - Windows Setup
echo ============================================================
echo.

echo Installing PyTorch CPU version...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other requirements...
pip install numpy==1.24.3
pip install matplotlib==3.7.1
pip install psutil==5.9.5
pip install tqdm==4.65.0
pip install pandas==2.0.2
pip install scikit-learn==1.2.2
pip install Pillow==9.5.0
pip install seaborn==0.12.2
pip install plotly==5.14.1

echo.
echo Creating necessary directories...
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo You can now run training with: python hrm_trainer.py
echo.
pause