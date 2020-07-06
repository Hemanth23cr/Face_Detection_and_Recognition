import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("C:/Users/heman/Desktop/speech/trainer.yml")
id = 0
font = cv2.FONT_HERSHEY_TRIPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if id == 1:
            id = "Hemanth"
        if id == 2:
            id = "Jayanthi"
        if id == 3:
            id = "Ramesh"
        if id == 4:
            id = "Aban"
        if id == 5:
            id = 'sanju'
        if id == 6:
            id = "akshay"
        cv2.putText(img, str(id), (x, y + h), font, 1,(255,255,0),1)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()