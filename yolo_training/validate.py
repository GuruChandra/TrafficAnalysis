from ultralytics import YOLO
import matplotlib.pyplot as plt

# ===========================
# Configuration
# ===========================
model_path = '../runs/traffic_detection6/weights/best.pt'  # Path to your best model
data_config = './data.yaml'

# ===========================
# Load the Model
# ===========================
model = YOLO(model_path)

# ===========================
# Run Validation
# ===========================
metrics = model.val(data=data_config, imgsz=640)

# ===========================
# Display Results
# ===========================
print("\n📊 **Validation Metrics**:")
print(f"mAP@0.5:    {metrics['metrics/mAP_0.5']}")
print(f"mAP@0.5:0.95: {metrics['metrics/mAP_0.5:0.95']}")
print(f"Precision:  {metrics['metrics/precision']}")
print(f"Recall:     {metrics['metrics/recall']}")
print(f"Speed:      {metrics['metrics/speed']} ms/image")

# Plot Confusion Matrix
model.plot_confusion_matrix()

# Plot Precision-Recall Curve
model.plot_pr_curve()

# Plot Validation Predictions
model.plot_results()
