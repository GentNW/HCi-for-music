#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install dollarpy


# In[2]:


import mediapipe as mp
import dollarpy as dp
import cv2
import socket
import pickle
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


# In[3]:


soc = socket.socket()
hostname = "localhost"# 127.0.0.1 #0.0.0.0
port = 65434
soc.bind((hostname,port))


# In[ ]:


soc.listen(5)
conn , addr = soc.accept()
print("device connected")


# In[ ]:


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    co = [results.pose_landmarks.landmarks[19].x,results.pose_landmarks.landmarks[19].y]
    co = str(co)
    msg=bytes(co,'utf-8')
    conn.send(msg)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


# In[ ]:


from dollarpy import Point , Recognizer , Template 
import os
import numpy as np


# In[ ]:


mp_holistic=mp.solutions.holistic


# In[ ]:


def getPoint(vidPath):
  with mp_hollistic.Holistic (static_image_mode=True ,
              min_detection_confidence=0.7,min_tracking_confidence=0.7) as holisitc :
    right_shoulder=[]
    left_shoulder=[]
    right_wrist=[]
    points=[]
    
    
    video = cv2.VideoCapture(0)
    while video.isOpened():
        ret, frame = video.read()

        # Recolor Feed
        if ret==True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            results = holisitc.process(cv2.flip(image,1))
            if results.pose_landmarks :
              right_hand.append(
                  Point(results.pose_landmarks.landmarks[19].x,
                        results.pose_landmarks.landmarks[17].y,
                        1)
              )
              left_hand.append(
                  Point(results.pose_landmarks.landmarks[20].x,
                        results.pose_landmarks.landmarks[18].y,
                        2)
              )
               
        else :
          break   
    points = right_hand+left_hand
    return points         


# In[ ]:


classes=[]


# In[ ]:


pth = ""#path of video 
points = getPoint(pth)
tmp = Template("",points)
classes.append(tmp)


# In[ ]:


pth = ""#path of video 
points = getPoint(pth)

import time 
start = time.time()
reco= Recognizer(classes)
results = reco.Recognize(points)
end=time.time()
duration = end-start
print (results)
print(duration)

