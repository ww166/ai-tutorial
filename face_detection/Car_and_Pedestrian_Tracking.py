import cv2

img_file = 'data/car-black.png'
classifier_file = 'data/haarcascade_car.xml'

# Create opencv image
img = cv2.imread(img_file)

# Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Convert to grayscale(needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars

car_coordinations = car_tracker.detectMultiScale(black_n_white, 1.21, 16)

# Draw rectangles around cars

for (x, y, w, h) in car_coordinations:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the car spotted

cv2.imshow('Car tracker', img)

cv2.waitKey()