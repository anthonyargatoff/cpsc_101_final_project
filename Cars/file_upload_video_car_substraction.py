# import required library
import cv2
import numpy as np

# Import the video
cap = cv2.VideoCapture('traffic.mp4')

kernel = None

backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = True)


while True:

    ret, frame = cap.read()
    

    if not ret:
        break

    foreground_mask = backgroundObject.apply(frame)


    _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
    

    foreground_mask = cv2.erode(foreground_mask, kernel, iterations = 1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations = 2)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()
    
    # loop over each contour found in the frame.
    for cnt in contours:

        # We need to be sure about the area of the contours i.e. it should be higher than 400 to reduce the noise.
        if cv2.contourArea(cnt) > 400:
            
            # Accessing the x, y and height, width of the cars
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Here we will be drawing the bounding box on the cars
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
            
            # Then with the help of putText method we will write the 'Car detected' on every car with a bounding box
            cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

            
    foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)
         
    stacked_frame = np.hstack((frame, foregroundPart, frameCopy))

    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked_frame, None, fx=1, fy=1)) # Can use stacked frame to view image proceesing

    
    k = cv2.waitKey(1) & 0xff
    
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

