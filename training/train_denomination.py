import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 224
DATA_DIR = "data"
CLASSES = ["10", "20", "50", "100", "200", "500"]

X, y = [], []

print("Loading denomination images...")

for idx, note in enumerate(CLASSES):
    for label in ["real", "fake"]:
        folder = os.path.join(DATA_DIR, note, label)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image)
            y.append(idx)

X = np.array(X) / 255.0
y = to_categorical(y, num_classes=len(CLASSES))

print("Total images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training denomination model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("Denomination Accuracy:", acc)

model.save("denomination_model.keras")
print("Model saved: denomination_model.keras")
