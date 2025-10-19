# Aditya's Improved Version
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
# Load and prepare the grayscale and color images
# --------------------------------------------------
gray_folder = './Data/Grayscale_2/'
color_folder = './Data/Colored_2/'

gray_files = sorted([f for f in os.listdir(gray_folder) if os.path.isfile(os.path.join(gray_folder, f))])
color_files = sorted([f for f in os.listdir(color_folder) if os.path.isfile(os.path.join(color_folder, f))])

common_files = [f for f in gray_files if f in color_files]
common_files = sorted(common_files)

images1 = []
images2 = []
for fname in common_files:
    gray_path = os.path.join(gray_folder, fname)
    color_path = os.path.join(color_folder, fname)

    # load color image and compute LAB
    color_img = load_img(color_path, target_size=(256,256))
    color_img = img_to_array(color_img).astype('float32') / 255.0
    lab_image = rgb2lab(color_img)

    # L in [0,1], ab in [-1,1]
    L = lab_image[:, :, 0] / 100.0
    ab = lab_image[:, :, 1:] / 128.0

    # Use the L from the color image (keeps pairing exact)
    images1.append(np.expand_dims(L, axis=-1))
    images2.append(ab)

# Convert to numpy arrays
X = np.array(images1, dtype=np.float32)
Y = np.array(images2, dtype=np.float32)

# Ensure correct shape for CNN input
if X.ndim == 3:
    X = np.expand_dims(X, axis=-1)

print("X dtype, min/max:", X.dtype, X.min(), X.max())
print("Y dtype, min/max:", Y.dtype, Y.min(), Y.max())
print("X shape, Y shape:", X.shape, Y.shape)

# --------------------------------------------------
# Define an improved CNN model
# --------------------------------------------------
x1 = keras.Input(shape=(256, 256, 1))

x2 = Conv2D(16, (3, 3), strides=2, padding='same')(x1)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)

x3 = Conv2D(32, (3, 3), strides=2, padding='same')(x2)
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)

x4 = Conv2D(64, (3, 3), strides=2, padding='same')(x3)
x4 = BatchNormalization()(x4)
x4 = Activation('relu')(x4)

x4b = Conv2D(128, (3, 3), padding='same')(x4)
x4b = BatchNormalization()(x4b)
x4b = Activation('relu')(x4b)

x5 = UpSampling2D((2, 2))(x4b)
x5 = Conv2D(32, (3, 3), padding='same', activation='relu')(x5)
x5 = BatchNormalization()(x5)

x6 = UpSampling2D((2, 2))(x5)
x6 = Conv2D(16, (3, 3), padding='same', activation='relu')(x6)
x6 = BatchNormalization()(x6)

x7 = UpSampling2D((2, 2))(x6)
x7 = Conv2D(2, (3, 3), activation='tanh', padding='same')(x7)

model = keras.Model(x1, x7)
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
model.summary()

# Split data before using the generator
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# --------------------------------------------------
# Train the model with validation and augmentation
# --------------------------------------------------
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.1)

callbacks = [
    keras.callbacks.ModelCheckpoint('best_model_faces.h5', save_best_only=True, monitor='val_loss'),
    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=32, shuffle=True, seed=seed),
    epochs=500,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=callbacks
)

# Evaluate on validation set
model.evaluate(X_val, Y_val, batch_size=2, verbose=2)

# Save model
model.save('model_faces_improved.h5')

# --------------------------------------------------
# Predict and visualize result
# --------------------------------------------------
folder_path = './Data/Grayscale_2/'
img = '1664.jpg'
img = folder_path + img

width, height = PILImage.open(img).size
print(width, height)

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

plt.subplot(1,2,1)
plt.imshow(img_resized, cmap='gray')
plt.title('Input (Grayscale)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(rgb_image)
plt.title('Colorized Output')
plt.axis('off')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
