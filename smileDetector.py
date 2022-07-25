# @author YD

import cv2
import numpy

faceDet = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileDet = cv2.CascadeClassifier('haarcascade_smile.xml')

camera = cv2.VideoCapture(0)
# or we can type the already captured video's file name instead of 0

while True:

    frameRead, frame = camera.read()

    if not frameRead:
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDet.detectMultiScale(frameGray)

    for(x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 300, 75), 3)

        face = frame[y: y + h, x: x + w]

        smiles = smileDet.detectMultiScale(frameGray, 1.5, 20)

        for (x1, y1, w1, h1) in face:

            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (50, 50, 100), 3)

    for(x, y, w, h) in smiles:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 100), 3)

    cv2.imshow('Smile detector', frame)

    cv2.waitKey()

camera.release()
cv2.destroyAllWindows()
