# EchoSense: Predictive Analysis of Cardiac Health

## Overview
Predictive Healthcare Analysis is an AI-driven system designed for ECG image classification using **EfficientNetB4** and **CNN models**. The project applies **deep learning** techniques to classify ECG images into different medical categories and provides a GUI for real-time predictions.

## Project Structure

```
Predictive-Healthcare-Analysis/
│── data_preprocessing.py       # Data augmentation & preprocessing
│── train_efficientnet.py       # EfficientNetB4-based ECG classifier
│── cnn_model.py                # Custom CNN model for ECG classification
│── ecg_image_loader.py         # Image loading & preprocessing
│── ecg_gui.py                  # GUI for ECG classification
│── ecg_model.h5                # Trained deep learning model
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
```

## Model Architecture
### **1. EfficientNetB4-based Model (`train_efficientnet.py`):**
- Uses **EfficientNetB4** as the backbone.
- Applies **Global Average Pooling (GAP)** instead of Flatten for better generalization.
- Includes **batch normalization & dropout** for regularization.
- Fine-tunes **30% of base model layers** for performance improvement.

### **2. Custom CNN Model (`cnn_model.py`):**
- Uses **Conv2D, MaxPooling, Flatten, Dense, Dropout** layers.
- Trained on ECG image dataset with **4 categories**.

### **3. Data Augmentation (`data_preprocessing.py`):**
- Uses **ImageDataGenerator** for on-the-fly augmentation.
- Performs **rotation, shifting, zooming, flipping, brightness adjustment**, etc.

### **4. Graphical User Interface (`ecg_gui.py`):**
- A **Tkinter-based GUI** for ECG classification.
- Allows users to **upload an ECG image** and receive a **predicted diagnosis**.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Predictive-Healthcare-Analysis.git
   cd Predictive-Healthcare-Analysis
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### **1. Train the Model**
   ```sh
   python train_efficientnet.py  # Train EfficientNetB4 model
   python cnn_model.py           # Train custom CNN model
   ```
### **2. Run the GUI for ECG Image Classification**
   ```sh
   python ecg_gui.py
   ```

## Dataset
- The dataset consists of **ECG images categorized into 4 classes**.
- Ensure your dataset is structured as follows:
  ```
  ECG Data/
  ├── Abnormal Heartbeat Patients/
  ├── Myocardial Infarction Patients/
  ├── Normal Person/
  ├── History of Myocardial Infarction/
  ```

## GAP Method in Model Architecture
The **Global Average Pooling (GAP)** layer is used instead of a **Flatten layer** because:
- It **reduces overfitting** by reducing the number of trainable parameters.
- It acts as a **structural regularizer**, improving generalization.
- GAP extracts **global spatial information** from feature maps.

## Future Improvements
- Expand dataset with more ECG images.
- Improve model accuracy with **hyperparameter tuning**.
- Implement **explainable AI (XAI)** techniques for medical transparency.

## Contributors
- **Meena Periasamy** (Northeastern University, M.S. Data Analytics)

## License
This project is licensed under the MIT License.

