from ultralytics import YOLO
import torch

# ===========================
# Configuration
# ===========================
model_name = 'yolov8s.pt'    # You can change to yolov8m.pt or yolov8l.pt for better models
data_config = './data.yaml'   # Path to the data.yaml
epochs = 5                  # Number of training epochs
img_size = 640                # Image size

# ===========================
# Model Training
# ===========================
model = YOLO(model_name)

if torch.backends.mps.is_available():
    print("✅ Training on Apple M1/M2 GPU (Metal Performance Shaders)")
    device = 'mps'
else:
    print("⚠️ MPS not available. Falling back to CPU.")
    device = 'cpu'


# ===========================
# Freeze all layers except the last one
# ===========================
freeze_layers = list(model.model.children())[:-1]  # All layers except the last one
for layer in freeze_layers:
    for param in layer.parameters():
        param.requires_grad = False

print("✅ All layers frozen except the final detection layer.")

model.train(
    data=data_config,
    epochs=epochs,
    imgsz=img_size,
    project='../runs',
    name='traffic_detection',
    workers=4,
    device=device
)
