import cv2
from PIL import Image
import time

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

for i in range(1, 1000):
    # Capture a single frame
    ret, frame = cap.read()
    # Create the filename
    filename = f'./Data/Colored_3/{i}.jpg'
    # Save the image as a file
    cv2.imwrite(filename, frame)
    # Add a small delay
    time.sleep(0.01)

# Release the video capture device
cap.release()

# Display the captured image
# cv2.imshow('Image', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
