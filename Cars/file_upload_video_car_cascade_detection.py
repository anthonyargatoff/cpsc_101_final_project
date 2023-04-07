# import required library
import cv2
import numpy as np


# Import video file
cap = cv2.VideoCapture('traffic.mp4')

# Import the cascade classifier
car_cascade_src = 'haarcascade_car.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)


while True:
    #reads frames from video
    ret, frames = cap.read()
    
    # Video processing to make easier for computer to make the proper detection
    
    # convert to grayscale of each frames
    grey = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    # Remove background noise
    blur = cv2.GaussianBlur(grey,(5,5),0)

    # Dilate the image
    dilated = cv2.dilate(blur,np.ones((3,3)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
         cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
         font = cv2.FONT_HERSHEY_DUPLEX
         cv2.putText(frames, 'Car', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
         
         # Display frames in a window
         cv2.imshow('Car Detection', frames)
         
         # Wait for Enter key to stop
    if cv2.waitKey(33) == 13:
        break

cap.release()
cv2.destroyAllWindows()