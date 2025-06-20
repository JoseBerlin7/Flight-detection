---

```markdown
# 🛰️ Drone Object Classification using YOLO Labels and TensorFlow CNN

This project focuses on classifying aerial objects such as **Drones**, **Birds**, **Helicopters**, and **Airplanes** from images using YOLO-annotated data and a Convolutional Neural Network built with TensorFlow.

## 📁 Dataset Structure

Ensure your dataset is structured like this:

```

Dataset/
└── tensorflow obj detection drone/
├── train/
│   ├── \_annotations.csv
│   ├── images/
│   └── labels/      <-- (auto-generated YOLO format)
└── test/
├── \_annotations.csv
├── images/
└── labels/      <-- (auto-generated YOLO format)

````

Each `_annotations.csv` should include the following columns:

- `filename`, `width`, `height`, `class`, `xmin`, `ymin`, `xmax`, `ymax`

## 🧠 Classes

```python
CLASS_MAP = {
  "Bird": 0,
  "Drone": 1,
  "AirPlane": 2,
  "Helicopter": 3
}
````

## ⚙️ Workflow Overview

1. **Convert** CSV annotations into YOLO format labels.
2. **Preprocess** images into TensorFlow-friendly tensors.
3. **Load** datasets and extract class IDs for classification.
4. **Build and train** a CNN model using TensorFlow.
5. **Track and log** experiments using MLflow.

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow pandas numpy opencv-python mlflow
```

> Ensure `mlflow` is properly set up and configured for experiment tracking.

### 2. Run the Training Script

```bash
python your_script_name.py
```

This will:

* Convert annotations to YOLO format
* Train a CNN model on the aerial object images
* Log parameters, metrics, and model artifacts to MLflow

### 3. Launch MLflow UI (Optional)

```bash
mlflow ui
```

Then visit `http://127.0.0.1:5000` in your browser to explore your experiment runs.

## 📊 Model

A simple ConvNet architecture:

```text
Conv2D(32) → MaxPool
Conv2D(64) → MaxPool
Conv2D(128) → MaxPool
Flatten → Dense(128) → Dense(NUM_CLASSES)
```

Loss: `sparse_categorical_crossentropy`
Optimizer: `Adam`

## 🧪 Notes

* Labels are parsed only for the **first object** per image.
* This is a **classification task**, not object detection.
* Be sure to fix image-label ordering before training!

## ✅ TODO (Future Improvements)

* Add bounding box regression (object detection)
* Use pre-trained models (MobileNet, ResNet)
* Add data augmentation
* Add early stopping and model checkpointing

---

## 🧠 Author

Built with 💻 and ☕ by Jose Berlin