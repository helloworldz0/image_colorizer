# Aditya's Improved Version
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
import win32gui
import win32con
import sys
import random
from PIL import Image as PILImage
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, Input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from sklearn.model_selection import train_test_split
from skimage.color import lab2rgb, rgb2lab
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# --------------------------------------------------
# Functions and Setup for GUI
# --------------------------------------------------
def on_enter_1(event):
    button1.config(bg="lightblue", fg="black")

def on_enter_2(event):
    button2.config(bg="lightblue", fg="black")

def on_enter_3(event):
    button3.config(bg="lightblue", fg="black")

def on_leave_1(event):
    button1.config(bg="#f0f0f0", fg="black")

def on_leave_2(event):
    button2.config(bg="#f0f0f0", fg="black")

def on_leave_3(event):
    button3.config(bg="#f0f0f0", fg="black")

root = tk.Tk()
root.title("My Simple GUI")
root.geometry("1280x720")

label = tk.Label(root, text="Image Colorizer", font=("Arial", 50))
label.pack(pady=25)

# --------------------------------------------------
# Predict and visualize result
# --------------------------------------------------
model = load_model('./best_model_faces.h5', compile=False)
# folder_path = './Data/Grayscale_People_Images/'
# img = '000000000077.jpg'
# img = folder_path + img

def on_run_click():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    # Capture a single frame
    ret, frame = cap.read()
    # Release the video capture device
    cap.release()

    # # Crop the image (you can adjust the coordinates)
    # x, y, w, h = 160, 120, 320, 240
    # cropped_frame = frame[y:y+h, x:x+w]

    cv2.imwrite('./temp.jpg', frame)
    img = './temp.jpg'

    # cv2.imwrite('./temp.jpg', cropped_frame)
    # img = './temp.jpg'

    width, height = PILImage.open(img).size
    print("Width, Height(Uncropped): ", width, height)

    # width, height = PILImage.open(img).size
    # print("Width, Height(Cropped): ", width, height)

    gray_img = load_img(img, color_mode='grayscale', target_size=(256,256))
    gray_img = img_to_array(gray_img).astype('float32') / 255.0
    X_test = np.reshape(gray_img, (1, 256, 256, 1))

    output = model.predict(X_test)
    output = np.reshape(output, (256, 256, 2))
    output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)

    outputLAB = np.zeros((height, width, 3), dtype=np.float32)
    img_resized = cv2.resize(np.reshape(gray_img, (256, 256)), (width, height), interpolation=cv2.INTER_CUBIC)

    # Denormalize: L*100, a,b*128
    outputLAB[:, :, 0] = img_resized * 100.0
    outputLAB[:, :, 1:] = output * 128.0

    # Clip to valid LAB ranges
    outputLAB[:, :, 0] = np.clip(outputLAB[:, :, 0], 0, 100)
    outputLAB[:, :, 1:] = np.clip(outputLAB[:, :, 1:], -128, 127)

    rgb_image = lab2rgb(outputLAB)

    plt.figure(num='Image Colorizer Output')
    plt.subplot(1,2,1)
    plt.imshow(img_resized, cmap='gray')
    plt.title('Input (Grayscale)')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rgb_image)
    plt.title('Colorized Output')
    plt.axis('off')
    plt.show()

def on_stop_click():
    plt.close('all')
    root.destroy()
    sys.exit("Terminating the program...")

def on_view_click():
    top_windows = []
    win32gui.EnumWindows(lambda hwnd, top_windows: top_windows.append((hwnd, win32gui.GetWindowText(hwnd))), top_windows)

    for hwnd, title in top_windows:
        if 'Image Colorizer Output'.lower() in title.lower():
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
            win32gui.SetForegroundWindow(hwnd)
            break

button1 = tk.Button(root, text="Run", command=on_run_click, width=25, height=5, font=("Arial", 15))
button1.pack(pady=25)

button2 = tk.Button(root, text="View", command=on_view_click, width=25, height=5, font=("Arial", 15))
button2.pack(pady=25)

button3 = tk.Button(root, text="Stop", command=on_stop_click, width=25, height=5, font=("Arial", 15))
button3.pack(pady=25)

button1.bind("<Enter>", on_enter_1)
button1.bind("<Leave>", on_leave_1)

button2.bind("<Enter>", on_enter_2)
button2.bind("<Leave>", on_leave_2)

button3.bind("<Enter>", on_enter_3)
button3.bind("<Leave>", on_leave_3)

root.mainloop()