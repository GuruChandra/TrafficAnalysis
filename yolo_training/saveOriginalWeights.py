import shutil
import os

# # Source path (default location)
# source = os.path.expanduser('~/.cache/ultralytics/yolov8s.pt')

# # Destination path
# destination = '../pretrained_models/yolov8s.pt'
# os.makedirs(os.path.dirname(destination), exist_ok=True)

# # Copy to the new path
# shutil.copy(source, destination)
# print(f"✅ YOLOv8s weights saved to {destination}")

from ultralytics import YOLO
import torch
import os

# Define your custom path
custom_path = '../runs/originalweigts/yolov8m.pt'

# Ensure the directory exists
os.makedirs(os.path.dirname(custom_path), exist_ok=True)

# Download the model to your custom path
torch.hub.download_url_to_file('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt', custom_path)

print(f"Model downloaded to {custom_path}")
model = YOLO(custom_path)
print(model)
print("✅ YOLOv8s weights downloaded to custom path!")
