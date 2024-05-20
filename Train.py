import keras.models
import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a simple dataset class with a constructor
class PyDataset:
    def __init__(self, data, labels, batch_size, augment=False, **kwargs):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self._num_samples = len(data)
        self.augment = augment
        self.dataset = self.create_dataset()

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.data, self.labels))
        dataset = dataset.shuffle(buffer_size=self._num_samples)
        if self.augment:
            dataset = dataset.map(self._augment_image)
        dataset = dataset.map(self._preprocess_image)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _preprocess_image(self, image, label):
        # Convert the image to grayscale
        #image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, (259, 259))
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
        return image, label

    def _augment_image(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        return image, label


# Function to load the dataset
def load_dataset(directory):
    images = []
    labels = []
    label_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4}  # Define label mapping
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        label_id = label_map[label]  # Map label string to integer label
        for file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, file)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(label_id)  # Use integer label
    return np.array(images), np.array(labels)


# Load the dataset
dataset_dir = r"D:\Downloads\COLLEGE\3RD YEAR\THSIS\try data\colored_images"
images, labels = load_dataset(dataset_dir)

# Split data into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

# Use PyDataset to wrap the training, validation, and testing data
train_dataset = PyDataset(train_images, train_labels, batch_size=32, augment=True)
val_dataset = PyDataset(val_images, val_labels, batch_size=32)
test_dataset = PyDataset(test_images, test_labels, batch_size=32)

# Define the number of output classes (assuming you have 5 classes including 'No_DR')
num_classes = 5

# Define the learning rate
learning_rate = 0.0001

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.Input(shape=(259, 259, 3)),  # Use Input layer instead of specifying input_shape
    tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model with the specified learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy since labels are integers
              metrics=['accuracy'])

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,  # Number of epochs to wait before stopping when no improvement is observed
    restore_best_weights=True  # Restore weights from the epoch with the best value of monitored quantity
)

# Train the model with data augmentation and validate on the validation set
history = model.fit(train_dataset.dataset, epochs=25, validation_data=val_dataset.dataset, callbacks=[early_stopping])

# Save the model along with optimizer and its state
model.save("DR_Algo_V1.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset.dataset)
print('Test accuracy:', test_acc)

model.summary()
