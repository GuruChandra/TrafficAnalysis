from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

# ===========================
# Configuration
# ===========================
model_path = '/app/pretained_models/yolov8s.pt'   # Update with your path
model = YOLO(model_path)
app = FastAPI()
output_folder = './results'
os.makedirs(output_folder, exist_ok=True)

# Initial load
if os.path.exists(model_path):
    model = YOLO(model_path)
    print("✅ Model Loaded Successfully from Docker Volume!")
else:
    model = None
    print("❌ Model not found. Please upload a valid model.")

# ===========================
# Serve HTML
# ===========================
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# ===========================
# Helper Functions
# ===========================
def reload_model():
    global model
    if os.path.exists(model_path):
        print("♻️  Reloading Model from disk...")
        model = YOLO(model_path)
        return {"status": "Model Reloaded Successfully"}
    else:
        model = None
        print("❌ Model file not found.")
        return {"status": "Model Reload Failed", "error": "File not found"}
    
def read_imagefile(file) -> np.ndarray:
    image = np.frombuffer(file, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def save_output(image, results):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path = os.path.join(output_folder, filename)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = box.conf[0]
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)
    return output_path

# ===========================
# API Endpoints
# ===========================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # ✅ Correctly await the read operation
        file_bytes = await file.read()
        
        # Read the image with OpenCV
        image = read_imagefile(file_bytes)
        
        if image is None:
            print("❌ Image read failed")
            return {"error": "Image could not be read"}
        
        # Run Inference
        print("🔍 Running Inference...")
        results = model.predict(source=image)
        
        # Save the output
        output_path = save_output(image, results)

        # Return the processed image
        return FileResponse(output_path, media_type="image/jpeg")
    
    except Exception as e:
        print(f"🔥 Error occurred during inference: {e}")
        return {"error": "Internal Server Error"}

@app.get("/reload/")
def reload():
    """ Manually Reload the Model """
    reload_model()
    return {"status": "Model Reloaded Successfully"}
