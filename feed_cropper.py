import cv2
from PIL import Image as PILImage

# Create a VideoCapture object
cap = cv2.VideoCapture(1)

# Capture a single frame
ret, frame = cap.read()

# Crop the image (you can adjust the coordinates)
x, y, w, h = 160, 120, 320, 240
cropped_frame = frame[y:y+h, x:x+w]

cv2.imwrite('./temp.jpg', frame)
img = './temp.jpg'

cv2.imwrite('./cropped_temp.jpg', cropped_frame)
cropped_img = './cropped_temp.jpg'

width, height = PILImage.open(img).size
print("Width, Height(Uncropped): ", width, height)

width, height = PILImage.open(cropped_img).size
print("Width, Height(Cropped): ", width, height)

# Release the video capture device
cap.release()

# Display the captured image
cv2.imshow('Cropped Image', cropped_frame)
cv2.imshow('Original Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
