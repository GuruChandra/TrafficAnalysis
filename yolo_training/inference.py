from ultralytics import YOLO
import cv2
import os

# ===========================
# Configuration
# ===========================
model_path = '../runs/traffic_detection6/weights/best.pt' #../runs/originalweigts/yolov8s.pt'  # Path to your original YOLOv8 weights
source = '../images'                            # Change to a single image path or directory
output_folder = './inference_results'
os.makedirs(output_folder, exist_ok=True)

# ===========================
# Load the Model
# ===========================
model = YOLO(model_path)
print(model.names)

# ===========================
# Run Inference
# ===========================
def run_inference(source):
    if os.path.isdir(source):
        images = [os.path.join(source, img) for img in os.listdir(source) if img.endswith(('.jpg', '.png'))]
    else:
        images = [source]

    for image_path in images:
        print(f"🔍 Processing: {image_path}")
        results = model.predict(source=image_path, save=False)#, save_txt=True, project=output_folder)
        
        for result in results:
            boxes = result.boxes
            img = cv2.imread(image_path)
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = box.conf[0]
                print('class id: ', class_id, "label", label, confidence)
                #if label in ['car','truck','traffic_light','bus'] and confidence > 0.2: 
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the output
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, img)
            print(f"✅ Saved Inference Result: {output_path}")

# ===========================
# Run
# ===========================
run_inference(source)
