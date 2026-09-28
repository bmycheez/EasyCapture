# EasyCapture — RAW Data Capture on Jetson + Sony IMX462

> Developed at **ENERZAi** (2023) to build low-light datasets for denoising research.

Capture RAW frames from a **Sony IMX462** sensor on **Jetson Xavier NX**, then convert them to RGB with a lightweight ISP.

## 🚀 Usage
```bash
conda create -n easy_capture python=3.8
conda activate easy_capture
pip install -r requirements.txt

# 1) Capture RAW frames
python3 run.py ./configs/jetson_imx462_raw_capture.py

# 2) RAW → RGB (naive ISP)
python3 raw2rgb.py {raw_path} [--show] [--save-dir DIR]

# 3) YUV → RGB
python3 yuv2rgb.py   # see script for options
```

## Role in the pipeline
```
EasyCapture (dataset) ─► model training ─► EasyISP (tuning) ─► EasyDemo (real-time demo)
```
