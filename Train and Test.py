import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import mlflow
import mlflow.keras

# Configurations and constants
CSV_DIR = "Dataset/tensorflow obj detection drone"
TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train", "_annotations.csv")
TEST_CSV_PATH = os.path.join(CSV_DIR, "test", "_annotations.csv")
TRAIN_LABELS_DIR = os.path.join(CSV_DIR, "train", "labels")
TEST_LABELS_DIR = os.path.join(CSV_DIR, "test", "labels")
TRAIN_IMAGES_DIR = os.path.join(CSV_DIR, "train", "images")
TEST_IMAGES_DIR = os.path.join(CSV_DIR, "test", "images")
CLASS_MAP = {"Bird": 0, "Drone": 1, "AirPlane": 2, "Helicopter": 3}
NUM_CLASSES = len(CLASS_MAP)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Helper Functions
def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def convert_bbox_to_yolo(xmin, xmax, ymin, ymax, img_width, img_height):
    center_x = ((xmin + xmax) / 2) / img_width
    center_y = ((ymin + ymax) / 2) / img_height
    box_width = (xmax - xmin) / img_width
    box_height = (ymax - ymin) / img_height
    return center_x, center_y, box_width, box_height

def csv_to_yolo(csv_path, labels_dir):
    clear_and_create_dir(labels_dir)
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        class_name = row['class']
        filename = row['filename']
        img_width = row['width']
        img_height = row['height']
        xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']
        class_id = CLASS_MAP[class_name]
        center_x, center_y, box_width, box_height = convert_bbox_to_yolo(xmin, xmax, ymin, ymax, img_width, img_height)
        yolo_annot = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"
        txt_file = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        with open(txt_file, 'a') as file:
            file.write(yolo_annot)
    print(f"Successfully converted annotations in {labels_dir}")

def parse_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:5])
            boxes.append((class_id, center_x, center_y, width, height))
    return boxes

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

def load_dataset(images_dir, labels_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    image_paths = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(images_dir, filename))
    
    def load_and_preprocess(path):
        return preprocess_image(path, target_size)
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_class_ids_from_yolo(labels_dir, image_names):
    class_ids = []
    for image_name in image_names:
        label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                if lines:
                    # Taking the first class ID in the file (for performing classification)
                    class_id = int(lines[0].strip().split()[0])
                    class_ids.append(class_id)
                else:
                    class_ids.append(-1)  # No label found
        else:
            class_ids.append(-1)  # No label file found
    return np.array(class_ids, dtype=int)


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Step 1: Converting the CSV annotations to YOLO format
    csv_to_yolo(TRAIN_CSV_PATH, TRAIN_LABELS_DIR)
    csv_to_yolo(TEST_CSV_PATH, TEST_LABELS_DIR)

    # Step 2: Loading images and Assigning the datasets to variables
    train_dataset = load_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    test_dataset = load_dataset(TEST_IMAGES_DIR, TEST_LABELS_DIR)

    # Step 3: Building the model and preparing REAL labels from YOLO files
    model = build_model((*IMAGE_SIZE, 3), NUM_CLASSES)

    # createing the list of image names
    train_image_names = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_image_names = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Getting the class IDs from YOLO label files
    train_class_ids = get_class_ids_from_yolo(TRAIN_LABELS_DIR, train_image_names)
    test_class_ids = get_class_ids_from_yolo(TEST_LABELS_DIR, test_image_names)

    # Checking for missing labels
    assert -1 not in train_class_ids, "Missing train labels detected!"
    assert -1 not in test_class_ids, "Missing test labels detected!"

    # Pairing images with their real labels
    train_dataset_with_labels = tf.data.Dataset.zip((
        train_dataset,
        tf.data.Dataset.from_tensor_slices(train_class_ids)
    ))
    test_dataset_with_labels = tf.data.Dataset.zip((
        test_dataset,
        tf.data.Dataset.from_tensor_slices(test_class_ids)
    ))

    # Step 4: MLflow experiment tracking
    mlflow.set_experiment("Drone_Detection")
    mlflow.tensorflow.autolog()  # Enabling autologging

    with mlflow.start_run():
        # parameter logging (to get some explicit control)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("num_classes", NUM_CLASSES)

        # Train model
        history = model.fit(
            train_dataset_with_labels,
            validation_data=test_dataset_with_labels,
            epochs=200,  
            verbose=1
        )

        # Manual model logging
        # mlflow.keras.log_model(model, "model")

    print("Training complete! Model and metrics logged to MLflow.")

if __name__ == "__main__":
    main()
