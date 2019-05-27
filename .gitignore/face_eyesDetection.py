@@ -0,0 +1,32 @@
import cv2 
import numpy as np 

face_xml = cv2.CascadeClassifier('face.xml')
eye_xml  = cv2.CascadeClassifier('eye.xml')

cap = cv2.VideoCapture(0)

while True :
    _ , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
    faces = face_xml.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[ x:x+w , y:y+h]
        roi_color = frame[x:x+w , y:y+h]

        eyes = eye_xml.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes :
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    cv2.imshow('faceDetection',frame)
    k = cv2.waitKey(27) & 0xff # Esc Key
    if k==27 :
        break

cap.release()  
cv2.destroyAllWindows()

     
