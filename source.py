from __future__ import print_function
import cv2 as cv
import argparse
import tkinter
from tkinter import messagebox
import logging
import datetime
import time 
from pygame import mixer  

mixer.init()

zero_count_faces = []
zero_count_eyes = []
count = 0

def show_alert(b, t):
    if(b == True and t == "face"):
        logging.warning('Watch out! FACE')
        mixer.music.load('face.mp3')
        mixer.music.play()
    elif (b == True and t == "eyes"):
        mixer.music.load('eyes.mp3')
        mixer.music.play()
        logging.warning('watch out, EYES')
    else:
        logging.warning('you are good')
    #alert_bool = True


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    
    if(len(zero_count_faces) > 50):
        zero_count_faces.pop(0)

    if(len(faces) < 1):
        zero_count_faces.append(0)
    else:
        zero_count_faces.append(1)

    
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)

        if(len(zero_count_eyes) > 50):
            zero_count_eyes.pop(0) 
        if(len(eyes) < 1):
            zero_count_eyes.append(0)
        else:
            zero_count_eyes.append(1)

        for (x2,y2,w2,h2) in eyes:

            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)


frame_face_count = 0
frame_eyes_count = 0

timer = datetime.datetime.now()

while True:

    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    face_count = 0
    eye_count = 0
    eye_percentage = 0
    face_percentage = 0
    detectAndDisplay(frame)
    
    for i in zero_count_faces:
        if(i == 0):
            face_count+=1
    for i in zero_count_eyes:
        if(i == 0):
            eye_count+=1
            
    
    face_percentage = face_count/50
    eye_percentage = eye_count/50

    if(face_percentage > .75):
        frame_face_count+=1
        if(frame_face_count > 30):
            show_alert(True, "face")
            frame_face_count = 0
    elif(eye_percentage <= 0.75):
        show_alert(False, "")

    if (face_percentage <= 0.75):
        if(eye_percentage > 0.75):
            frame_eyes_count += 1
            if (frame_eyes_count > 30):
                show_alert(True, "eyes")
                frame_eyes_count = 0
        else:
            show_alert(False, "")

    if cv.waitKey(10) == 27:
        break
    