from aimodule import Aimodel as Mod
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from PIL import Image as PILImage , ImageQt
import matplotlib.pyplot as plt

model=Mod('./tests.keras')
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
img='./temp.jpg'
cv2.imwrite(img, frame)
# cv2.imshow("dis",frame)
model.bw(img)

colorized=model.predict()
print(colorized.shape)

plt.subplot(1,2,1)
plt.imshow(model.gray_img, cmap='gray')
plt.title('Input (Grayscale)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(colorized)
plt.title('Colorized Output')
plt.axis('off')
plt.show()