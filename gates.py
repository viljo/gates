import time
import cv2
from imutils.video import FileVideoStream
import imutils
import numpy as np
from scipy import ndimage

fvs1 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/101/").start()

#mask1 = np.array([[285,162],[375,101],[466,113],[477,226],[304,282]], np.int32) # polygon för hela grinden inklusive topp
#mask1 = mask1.reshape((-1,1,2))

# prepp för perspektiv algoritm
pts1 = np.float32([[285,162],[467,113],[304,282],[477,225]]) # Grinden TL, TR, BL, BR
pts2 = np.float32([[0,0],[250,0],[0,200],[250,200]])
M = cv2.getPerspectiveTransform(pts1,pts2)

oldtime = time.time()
index = 0
white_percent_old = 0
status = "none"
while(1):
    frame1 = fvs1.read()

    if np.shape(frame1) == (): # check för empty frame
        continue

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.warpPerspective(frame1,M,(250,200))
    frame1 = cv2.copyMakeBorder(frame1, 3, 3, 3, 3, cv2.BORDER_CONSTANT, 1) # make black frame around image

    frame1 = cv2.adaptiveThreshold(frame1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    vertical = cv2.bitwise_not(np.copy(frame1))
    rows = vertical.shape[0]
    vertical_size = rows // 3
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical_size = rows
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical = cv2.dilate(vertical, horizontalStructure)

    vertical = cv2.bitwise_not(np.copy(vertical))
    white_pixels = cv2.countNonZero(vertical)

    white_percent = 0
    if white_pixels > 0:
        white_percent= int((white_pixels/32000)*100)

    #run every seconds

    if time.time() - oldtime > 1:
        oldtime = time.time()
        if white_percent_old == white_percent:
            if white_percent == 0:
                status = "closed" 
            else:
                status = "open" 
        elif white_percent_old > white_percent:
            status = "closing"   
        elif white_percent_old < white_percent:
            status = "opening"  
        white_percent_old=white_percent

    position = (10,50)
    cv2.putText(
        vertical, #numpy array on which text is written
        (str(white_percent) + " % (" + status +")"), #text
        position, #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        1, #font size
        (209, 80, 0, 255), #font color
        3) #font stroke

    cv2.imshow('VIDEO1_vertical', vertical)

    # rita ut grinden på bilden (enast för hel otransformerad bild)
    #cv2.polylines(frame1,[mask1],True,(0,255,0),1)
    
    # Maska ut grinden (enast för hel otransformerad bild)
    #stencil = np.zeros(frame1.shape).astype(frame1.dtype)
    #cv2.fillPoly(stencil, [mask1], [255, 255, 255])
    #frame1 = cv2.bitwise_and(frame1, stencil)
    #Klipp ut grinden rektangulärt (Endast för hel otransformerad bild)
    #frame1 = frame1[101:285, 282:477]

    #frame1 = ndimage.rotate(frame1, -18)

    cv2.imshow('VIDEO1', frame1)

    cv2.waitKey(1)
