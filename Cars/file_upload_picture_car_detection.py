# import required library
import cv2

# read the input image
img = cv2.imread('cars2.png')

# Below is image processing to make it easier for the computer to recognize faces (takes less resources I believe)

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read haacascade to detect faces in input image
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# detects faces in the input image
cars = car_cascade.detectMultiScale(gray, 1.1, 2)
print('Number of detected cars:', len(cars))\

# loop over all the detected faces
for (x,y,w,h) in cars:

   # To draw a rectangle around the detected face  
   cv2.rectangle(img,(x,y),(x+w,y+h),(57,255,51),2)

# Resize the image
resize = cv2.resize(img, (960, 540))

# Display an image in a window
cv2.imshow('Car Detection', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()