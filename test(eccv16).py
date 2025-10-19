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
import torch
from torchvision import transforms
from colorizers import eccv16
from PIL import Image as PILImage
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
# Load the Pretrained ECCV 2016 Model
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading ECCV 2016 pretrained model...")
model = eccv16(pretrained=True).to(device).eval()
print("Model loaded successfully!")
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

    if not ret:
        print("❌ Failed to capture image from webcam.")
        return

    # # Crop the image (you can adjust the coordinates)
    # x, y, w, h = 160, 120, 320, 240
    # cropped_frame = frame[y:y+h, x:x+w]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = PILImage.fromarray(frame_rgb)

    # cv2.imwrite('./temp.jpg', cropped_frame)
    # img = './temp.jpg'

    width, height = img_rgb.size
    print("Width, Height(Uncropped): ", width, height)

    # width, height = PILImage.open(img).size
    # print("Width, Height(Cropped): ", width, height)

    # Convert the image to grayscale (L channel)
    # Convert to L channel in [0,100]
    img_lab = rgb2lab(np.array(img_rgb))
    img_l = img_lab[:, :, 0]

    # Resize for model input
    img_l_rs = cv2.resize(img_l, (256, 256))
    tens_l = torch.from_numpy(img_l_rs).unsqueeze(0).unsqueeze(0).float().to(device)
    # ❌ DO NOT divide or subtract 50 — model does it internally

    # Predict ab channels
    with torch.no_grad():
        output_ab = model(tens_l)

    output_ab = output_ab.cpu().numpy()[0].transpose((1, 2, 0))
    output_ab = cv2.resize(output_ab, (width, height))

    # Combine L and ab to form LAB image
    outputLAB = np.zeros((height, width, 3), dtype=np.float32)
    outputLAB[:, :, 0] = img_l
    outputLAB[:, :, 1:] = output_ab   # already scaled correctly

    # Clip and convert
    outputLAB[:, :, 0] = np.clip(outputLAB[:, :, 0], 0, 100)
    outputLAB[:, :, 1:] = np.clip(outputLAB[:, :, 1:], -128, 127)

    rgb_image = np.clip(lab2rgb(outputLAB), 0, 1)

    # Display the grayscale and colorized images
    plt.figure(num='Image Colorizer Output')
    plt.subplot(1,2,1)
    plt.imshow(img_l / 100.0, cmap='gray')
    plt.title('Input (Grayscale)')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rgb_image)
    plt.title('Colorized Output (ECCV 2016)')
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