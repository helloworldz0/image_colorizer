import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Capture a single frame
ret, frame = cap.read()

# Release the video capture device
cap.release()

# Display the captured image
cv2.imshow('Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
