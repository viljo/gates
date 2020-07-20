import cv2
from imutils.video import FileVideoStream
import imutils
import numpy as np

fvs1 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/101/").start()
fvs2 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/201/").start()

while(1):

    frame1 = fvs1.read()
    frame1 = frame1[90:320, 240:540]
    cv2.imshow('VIDEO1', frame1)

    frame2 = fvs2.read()
    frame2 = frame2[100:300, 650:900]
    cv2.imshow('VIDEO2', frame2)

    cv2.waitKey(1)
