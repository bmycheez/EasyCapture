# Install
```
conda crate -n easy_capture python=3.8
conda activate easy_capture
pip install -r requirements.txt
```
<br>
<br>

# Capture RAW Data from IMX462
```
python3 run.py ./configs/jetson_imx462_raw_capture.py
```
<br>
<br>

# RAW Image to RGB (Naive ISP)
```
python3 raw2rgb.py {raw_path} {--show} {--save-dir ...}
```