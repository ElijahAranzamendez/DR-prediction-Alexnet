import numpy as np
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
from tkinterdnd2 import DND_FILES, TkinterDnD

# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path) #cv2.IMREAD_GRAYSCALE
    image = cv2.resize(image, (259, 259))  # Resize image to match input size of the model
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    return image

# Function to predict DR stage
def predict_DR_stage(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    confidence_score = prediction[0][predicted_class]  # Confidence score of the predicted class
    return predicted_class, confidence_score

# Load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model('DR_Algo_V1.h5')
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error loading the model: {e}")
        return None

# Function to open file dialog and select an image
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        load_image(file_path)

# Function to load image into the GUI
def load_image(file_path):
    global image_path
    image_path = file_path
    img = Image.open(file_path)
    img = ImageOps.contain(img, (259, 259), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    prediction_label.configure(text="")  # Clear previous prediction
    confidence_label.configure(text="")

# Function to handle prediction button click
def predict():
    global image_path
    if not image_path:
        messagebox.showwarning("Warning", "Please load an image first")
        return
    try:
        prediction, confidence_score = predict_DR_stage(image_path, model)
        display_DR_stage(prediction, confidence_score)
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred during prediction: {e}")

# Function to display the predicted DR stage along with confidence score
def display_DR_stage(prediction, confidence_score):
    classes = ['0: No_DR', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferate_DR']
    prediction_label.configure(text=f"Predicted DR stage: {classes[prediction]}")
    confidence_label.configure(text=f"Confidence score: {confidence_score}")

# Function to handle drag-and-drop of files
def drop(event):
    file_path = event.data
    if file_path:
        file_path = file_path.strip('{}')  # Remove curly braces if present
        load_image(file_path)

# Initialize the main window
root = TkinterDnD.Tk()
root.title("DR Stage Detection")
root.geometry("400x500")

# Load the trained model
model = load_model()

# Create and place the image panel
panel = ttk.Label(root)
panel.pack(pady=20)

# Create and place buttons
browse_button = ttk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

predict_button = ttk.Button(root, text="Predict DR Stage", command=predict)
predict_button.pack(pady=10)

# Create labels for prediction results
prediction_label = ttk.Label(root, text="")
prediction_label.pack(pady=10)

confidence_label = ttk.Label(root, text="")
confidence_label.pack(pady=5)

# Bind the drag-and-drop event to the root window
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

# Global variable to store image path
image_path = None

# Start the main loop
root.mainloop()
