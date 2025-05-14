
import os
import cv2
import glob
import random
import shutil
import albumentations as A
from tqdm import tqdm
from xml.etree import ElementTree as ET

# ==============================
# Configuration
# ==============================
DATASET_PATH = '/Users/guruchandrav/Documents/WorkSpace/Datasets/openimages_data/'
OUTPUT_PATH = './preprocessed_data'
TRAIN_SPLIT = 0.8
IMG_SIZE = 640
AUGMENT = True

# ==============================
# Augmentation Strategy
# ==============================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0)  # Resize instead of crop
    #A.RandomScale(scale_limit=0.1, p=0.5),
    #A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE, p=0.2)
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))


# ==============================
# Create Directories
# ==============================
os.makedirs(f'{OUTPUT_PATH}/train/images', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/train/labels', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/val/images', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/val/labels', exist_ok=True)

# ==============================
# Helper Functions
# ==============================
def parse_xml(xml_file, original_width, original_height):
    objects = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels=[]
    bboxes=[]
    scale_x = IMG_SIZE / original_width
    scale_y = IMG_SIZE / original_height
    class_mapping ={
        "car": 0,
        "bus":1,
        "traffic_light":2,
        "traffic light":2
    }
    for obj in root.iter('object'):
        label = obj.find('name').text
        if label == 'traffic light': 
            label = 'traffic_light'
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text) * scale_x
        ymin = float(xmlbox.find('ymin').text) * scale_y
        xmax = float(xmlbox.find('xmax').text) * scale_x
        ymax = float(xmlbox.find('ymax').text) * scale_y

        x_center = ((xmin + xmax) / 2) / IMG_SIZE
        y_center = ((ymin + ymax) / 2) / IMG_SIZE
        bbox_width = (xmax - xmin) / IMG_SIZE
        bbox_height = (ymax - ymin) / IMG_SIZE

        #objects.append(f"{label} {x_center} {y_center} {bbox_width} {bbox_height}")
        labels.append(class_mapping[label])
        bboxes.append([x_center,y_center,bbox_width,bbox_height])
    return bboxes,labels

# ==============================
# Process Data
# ==============================
def process_images():
    categories = ['traffic_light', 'bus', 'car']
    images = []

    for category in categories:
        images += glob.glob(f'{DATASET_PATH}/{category}/images/*.png') + glob.glob(f'{DATASET_PATH}/{category}/images/*.jpg')
    
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for phase, image_set in [('train', train_images), ('val', val_images)]:
        for img_path in tqdm(image_set, desc=f'Processing {phase} images'):
            img = cv2.imread(img_path)
            filename = os.path.basename(img_path)
            category = img_path.split('/')[-3]
            xml_path = f"{DATASET_PATH}/{category}/pascal/{filename.replace('.jpg', '.xml').replace('.png', '.xml')}"

            # Get original size before resizing
            original_height, original_width = img.shape[:2]

            # Resize the image to a fixed size before augmentation
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Parse the bounding boxes
            bboxes, labels = parse_xml(xml_path, original_width, original_height)
            # Perform augmentation
            if AUGMENT and phase == 'train':
                #print('Bboxes:', bboxes)
                #print("Labels:", labels)
                augmented = transform(image=img, bboxes=bboxes, category_ids=labels)
                img = augmented['image']
                bboxes = augmented['bboxes']

            #print("filename:",filename)
            cv2.imwrite(f'{OUTPUT_PATH}/{phase}/images/{filename}', img)
            # Write the new labels
            if len(bboxes) > 0:
                label_file = f"{OUTPUT_PATH}/{phase}/labels/{filename.replace('.jpg', '.txt').replace('.png', '.txt')}"
                with open(label_file, 'w') as f:
                    for bbox, label in zip(bboxes, labels):
                        f.write(f"{label} {' '.join(map(str, bbox))}\n")

    print(f"✅ Data processing complete. Check the '{OUTPUT_PATH}' folder.")

if __name__ == "__main__":
    process_images()
