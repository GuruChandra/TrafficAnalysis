import cv2
import os
import random
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

def display_image_with_bbox_comparison(output_path='./preprocessed_data', dataset_path='/Users/guruchandrav/Documents/WorkSpace/Datasets/openimages_data/'):
    class_mapping ={
        '0':"car",
        '1':"bus",
        '2':"traffic_light"
    }
    for _ in range(10):
        # Select train or val folder randomly
        phase = random.choice(['train', 'val'])
        
        # Get a random image from the chosen phase
        image_dir = os.path.join(output_path, phase, 'images')
        label_dir = os.path.join(output_path, phase, 'labels')
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print("No images found in the specified directory.")
            return

        random_image = random.choice(image_files)
        image_path = os.path.join(image_dir, random_image)
        label_path = os.path.join(label_dir, random_image.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load image
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Plot 1: YOLO BBoxes
        yolo_img = img.copy()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    
                    label_data = line.strip().split()
                    #print(label_data)
                    label = label_data[0]
                    print(label_data)
                    x_center, y_center, bbox_width, bbox_height = map(float, label_data[1:])

                    # Convert normalized coordinates to image coordinates
                    x_center = int(x_center * width)
                    y_center = int(y_center * height)
                    bbox_width = int(bbox_width * width)
                    bbox_height = int(bbox_height * height)

                    # Calculate the top-left corner of the bounding box
                    xmin = int(x_center - bbox_width / 2)
                    ymin = int(y_center - bbox_height / 2)
                    xmax = int(x_center + bbox_width / 2)
                    ymax = int(y_center + bbox_height / 2)
                    label = class_mapping[label]
                    # Draw bounding box and label
                    cv2.rectangle(yolo_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(yolo_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Plot 2: Original Pascal VOC BBoxes
        xml_img_path = os.path.join(dataset_path, label, 'images', random_image)
        xml_img = cv2.imread(xml_img_path)#img.copy()
        category = label
        
        #print("categorgy:", category)
        xml_path = os.path.join(dataset_path, category, 'pascal', random_image.replace('.jpg', '.xml').replace('.png', '.xml'))
        #print('xml path: ',xml_path)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.iter('object'):
                label = obj.find('name').text
                xmlbox = obj.find('bndbox')
                xmin = int(xmlbox.find('xmin').text)
                ymin = int(xmlbox.find('ymin').text)
                xmax = int(xmlbox.find('xmax').text)
                ymax = int(xmlbox.find('ymax').text)

                # Draw bounding box and label
                cv2.rectangle(xml_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(xml_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display both images for comparison
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))
        plt.title(f'YOLO BBoxes - {random_image}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(xml_img, cv2.COLOR_BGR2RGB))
        plt.title(f'XML BBoxes - {random_image}')
        plt.axis('off')

        plt.show()

# Call the function
display_image_with_bbox_comparison()
