import tensorflow as tf
import os
import cv2
import numpy as np
import random
from PIL import Image as PILImage
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, Input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from sklearn.model_selection import train_test_split
from skimage.color import lab2rgb, rgb2lab
# from tk import *
class Aimodel:
    def __init__(self,modelpath):
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        self.model = load_model(modelpath, compile=False)
    def bw(self,imgpath):
        self.gray_img = load_img(imgpath, color_mode='grayscale', target_size=(256,256))
        self.gray_img = img_to_array(self.gray_img).astype('float32') / 255.0
        self.X_test = np.reshape(self.gray_img, (1, 256, 256, 1))

    def predict(self):
        model = self.model
        width, height = 640,480

        output = model.predict(self.X_test)
        output = np.reshape(output, (256, 256, 2))
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
        outputLAB = np.zeros((height, width, 3), dtype=np.float32)
        img_resized = cv2.resize(np.reshape(self.gray_img, (256, 256)), (width, height), interpolation=cv2.INTER_CUBIC)
        outputLAB[:, :, 0] = img_resized * 100.0
        outputLAB[:, :, 1:] = output * 128.0

        outputLAB[:, :, 0] = np.clip(outputLAB[:, :, 0], 0, 100)
        outputLAB[:, :, 1:] = np.clip(outputLAB[:, :, 1:], -128, 127)

        rgb_image = lab2rgb(outputLAB)
        return rgb_image