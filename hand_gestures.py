import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# === Unzip Dataset ===
zip_path = "archive.zip"
extract_path = "gesture_dataset"

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Dataset already extracted.")

# === Print structure to understand layout ===
def print_folder_structure(base_path):
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:5]:  # Show only 5 sample files per folder
            print(f"{' ' * 2 * (level + 1)}{f}")

print("\n Dataset Structure:")
print_folder_structure(extract_path)

# === Configure dataset path ===

dataset_path = extract_path  # may need to change this based on actual layout

# ===  Prepare image data generators ===
image_size = (160, 160)
batch_size = 32
epochs = 10

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

# ===  Build model using MobileNetV2 ===
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
base_model.trainable = False  # Freeze the base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===  Train the model ===
print("\n Starting Training...\n")
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# ===  Save the model ===
model.save("gesture_model_mobilenetv2.h5")
print("\n Model saved as gesture_model_mobilenetv2.h5")

# ===  Plot accuracy graph ===
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("training_plot.png")
plt.show()