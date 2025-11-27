import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# --- Pfade ---
DATASET_PATH = "/home/berat/twinsecure/dataset_faces"
EXPORT_PATH = "/home/berat/twinsecure/models/model_float.tflite"
LABELS_PATH = "/home/berat/twinsecure/models/labels.txt"

IMG_SIZE = (224, 224)

def load_data():
    images = []
    labels = []
    label_names = sorted(os.listdir(DATASET_PATH))
    with open(LABELS_PATH, "w") as f:
        for idx, name in enumerate(label_names):
            f.write(name + "\n")
            folder = os.path.join(DATASET_PATH, name)
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    path = os.path.join(folder, file)
                    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                    images.append(np.array(img))
                    labels.append(idx)
    return np.array(images), np.array(labels)

def build_model(num_classes):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("Lade Daten…")
    X, y = load_data()
    print(f"{len(X)} Bilder geladen")

    model = build_model(num_classes=len(set(y)))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Trainiere Modell…")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)

    print("Exportiere als TFLite…")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(EXPORT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Modell gespeichert unter: {EXPORT_PATH}")
    print("Jetzt mit edgetpu_compiler kompilieren für Coral.")

if __name__ == "__main__":
    main()
