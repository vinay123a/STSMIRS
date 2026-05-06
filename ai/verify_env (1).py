"""Quick environment verification script for STSMIRS."""
import os
import sys

if not os.environ.get("YOLO_CONFIG_DIR"):
    os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")

print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

import flask
print(f"Flask: {flask.__version__}")

import sklearn
print(f"scikit-learn: {sklearn.__version__}")

import numpy
print(f"NumPy: {numpy.__version__}")

import matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

print("\n--- ALL IMPORTS OK ---")
