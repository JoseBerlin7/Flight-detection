# I'm gonna use YoloV8 for this model

# For this dataset I'm planning to perform 
# 1. Creating txt file for each image named same as  image file.
# 2. Creating labels for each images which would make the process easier in YOLO to identify the image where YOLO expects the following format class_id center_x center_y width height
''' Creating Labels
class_id: {drone:1, bird :0, ..}
center_x = (xmin + xmax) / 2 / width
center_y = (ymin + ymax) / 2 / height
width = (xmax - xmin) / width
height = (ymax - ymin) / height
'''
 
import os
import pandas as pd

# Converting the CSV to the (class_id center_x center_y width height) format to meet YOLO model desired format
class_map = { "Bird":0, "Drone":1,"AirPlane":2, "Helicopter":3} 

# To convert the csv we have to the yolo format
def csv_2_yolo(csv_path, store_dir):
    
    data = pd.read_csv(csv_path)
    os.makedirs(store_dir, exist_ok= True)

    for _, row in data.iterrows():
        class_name = row['class']
        filename = row['filename']
        img_width = row['width']
        img_height = row['height']
        xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']

        # YOLO Format
        class_id = class_map[class_name]
        center_x = ((xmin+xmax)/2)/img_width
        center_y = ((ymin+ymax)/2)/ img_height
        box_width = (xmax-xmin)/img_width
        box_height =(ymax-ymin)/img_height

        # YOLO annotation
        yolo_annot = f"{class_id} {center_x} {center_y} {box_width} {box_height}"

        # Saving in a text file 
        txt_file = os.path.join(store_dir, os.path.splitext(filename)[0]+".txt")
        with open(txt_file, 'a') as file:
            file.write(yolo_annot)
        
    print(f"Success on {store_dir}")

train_csv_dir = "Dataset/tensorflow obj detection drone/train/_annotations.csv"
train_output_dir = "Dataset/tensorflow obj detection drone/train/labels"

csv_2_yolo(train_csv_dir, train_output_dir)

test_csv_dir = "Dataset/tensorflow obj detection drone/test/_annotations.csv"
test_output_dir = "Dataset/tensorflow obj detection drone/test/labels"

csv_2_yolo(test_csv_dir, test_output_dir)