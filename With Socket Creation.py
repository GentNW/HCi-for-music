#!/usr/bin/env python
# coding: utf-8

# In[1]:
! pip install dollarpy


# In[2]:
from dollarpy import Point , Recognizer , Template 

import os
import numpy as np
import mediapipe as mp
import dollarpy as dp
import cv2
import socket
import pickle
import json

# In[3]:

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

soc = socket.socket()
hostname = "localhost"# 127.0.0.1 #0.0.0.0
port = 65434
soc.bind((hostname,port))


# In[ ]:
#soc.listen(5)
#conn , addr = soc.accept()
#print("device connected")

# In[ ]:

#testing the classification live

def Classify(testpoint):
  reco= Recognizer(classes)
  print(reco)
  if reco == None:
        return 0
  else:
    
    results = reco.recognize(testpoint)
  return results
#pth = ""#path of video 
#points = getPoint(pth)

#import time 
#start = time.time()
#reco= Recognizer(classes)
#results = reco.Recognize(points)
#end=time.time()
#duration = end-start
#print (results)
#print(duration)

# In[ ]:
def live_cam():
  right_shoulder=[]
  left_shoulder=[]
  points=[]
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
      if results.pose_landmarks :
                right_shoulder.append(
                    Point(results.pose_landmarks.landmark[19].x,
                          results.pose_landmarks.landmark[17].y,
                          1)
                )
                left_shoulder.append(
                    Point(results.pose_landmarks.landmark[20].x,
                          results.pose_landmarks.landmark[18].y,
                          2)
                )
      points = right_shoulder+left_shoulder
      if len(points) >= 150 and len(points) < len(classes):
                      right_shoulder = []
                      left_shoulder = []
                      res = Classify(points)
                      print(res[0])
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      #co = [results.pose_landmarks.landmarks[19].x,results.pose_landmarks.landmarks[19].y]
      #co = str(co)
      #msg=bytes(co,'utf-8')
      #conn.send(msg)
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyallwindows()
        break
  cv2.destroyallwindows()
  cap.release()
# In[ ]:
def getPoint(vidPath):
  with mp_pose.Pose (min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose :
    right_shoulder=[]
    left_shoulder=[]
    right_wrist=[]
    points=[]
    
    
    video = cv2.VideoCapture(vidPath)
    while video.isOpened():
        ret, frame = video.read()

        # Recolor Feed
        if ret==True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            results = pose.process(cv2.flip(image,1))
            if results.pose_landmarks :
              right_shoulder.append(
                  Point(results.pose_landmarks.landmark[19].x,
                        results.pose_landmarks.landmark[17].y,
                        1)
              )
              left_shoulder.append(
                  Point(results.pose_landmarks.landmark[20].x,
                        results.pose_landmarks.landmark[18].y,
                        2)
              )
               
        else :
          break   
    points = right_shoulder+left_shoulder
    return points

# In[ ]:
classes=[]

#Setting up the templates for classification

pth = "Video/trim 1"#path of video 
points = getPoint(pth)
tmp = Template("Open",points)
classes.append(tmp)

pth = "Video/trim 2"#path of video 
points = getPoint(pth)
tmp = Template("Open",points)
classes.append(tmp)

pth = "Video/trim 3"#path of video 
points = getPoint(pth)
tmp = Template("Open",points)
classes.append(tmp)

pth = "Video/trim 4"#path of video 
points = getPoint(pth)
tmp = Template("Open",points)
classes.append(tmp)

pth = "Video/trim 5"#path of video 
points = getPoint(pth)
tmp = Template("Open",points)
classes.append(tmp)

pth = "Video/trim 6"#path of video 
points = getPoint(pth)
tmp = Template("Close",points)
classes.append(tmp)

pth = "Video/trim 7"#path of video 
points = getPoint(pth)
tmp = Template("Close",points)
classes.append(tmp)

pth = "Video/trim 8"#path of video 
points = getPoint(pth)
tmp = Template("Close",points)
classes.append(tmp)

pth = "Video/trim 9"#path of video 
points = getPoint(pth)
tmp = Template("Close",points)
classes.append(tmp)

pth = "Video/trim 10"#path of video 
points = getPoint(pth)
tmp = Template("Close",points)
classes.append(tmp)
# In[ ]:
live_cam()
# In[ ]:
print("saba7o")
