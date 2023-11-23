import cv2

img_file = 'bmw.jpg'
classifier_file = 'car_detector.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the faces spotted
cv2.imshow('Paul Diwkar Car Detector', img)

# Dont autoclose (wait here in the code and listen for a key press)
cv2.waitKey()

print ("Code Completed")