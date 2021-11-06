import cv2

# Load some pre-trained data on face frontals from openvcv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


# To capture video from webcam
web_cam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    successful_frame_read, frame = web_cam.read()
    
    # Must convert to grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img, 1.0485258, 6)

    print(face_coordinates)
    # [[282 161 217 217]]

    # Draw rectangle around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('Face Detector', frame)

    # 延迟关闭cv2窗口
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 113 or key == 81:
        break

web_cam.release()



"""

img = cv2.imread('data/RT.png')

# Must convert to grayscale
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

print(face_coordinates)
# [[282 161 217 217]]

# Draw rectangle around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(gray_scaled_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow('Face Detector', gray_scaled_img)

# 延迟关闭cv2窗口
cv2.waitKey()

"""