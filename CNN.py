import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_images_from_directory(directory, class_labels):
    images, labels = [], []
    for label, index in class_labels.items():
        folder = os.path.join(directory, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                images.append(image)
                labels.append(index)
    return np.array(images), np.array(labels)

class_map = {
    'Abnormal Heartbeat Patients': 0,
    'Myocardial Infarction Patients': 1,
    'Normal Person': 2,
    'Patient that have History of Myocardial Infraction': 3
}

dataset_path = 'Your directory'
images, labels = load_images_from_directory(dataset_path, class_map)
images = images.astype('float32') / 255.0
labels = to_categorical(labels, num_classes=len(class_map))

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('ecg_model.h5')