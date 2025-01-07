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
 