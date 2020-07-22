import cv2
from imutils.video import FileVideoStream
import imutils
import numpy as np
from scipy import ndimage

def _find_edges_laplacian(image, edge_multiplier):
    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray[..., 0]
    norm_image = cv2.normalize(image_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    edges_f = cv2.Laplacian(norm_image, cv2.CV_64F)
    edges_f = np.abs(edges_f)
    edges_f = edges_f ** 2
    vmax = np.percentile(edges_f, min(int(90 * (1/edge_multiplier)), 99))
    edges_f = np.clip(edges_f, 0.0, vmax) / vmax

    edges_uint8 = np.clip(np.round(edges_f * 255), 0, 255.0).astype(np.uint8)
    edges_uint8 = cv2.medianBlur(edges_uint8, 3)
    edges_uint8 = cv2.adaptiveThreshold(edges_uint8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return edges_uint8

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

fvs1 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/101/").start()
fvs2 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/201/").start()

#mask1 = np.array([[285,162],[375,101],[466,113],[477,226],[304,282]], np.int32) # polygon för hela grinden inklusive topp
#mask1 = mask1.reshape((-1,1,2))

# prepp för perspektiv algoritm
pts1 = np.float32([[285,162],[467,113],[304,282],[477,225]]) # Grinden TL, TR, BL, BR
pts2 = np.float32([[0,0],[250,0],[0,200],[250,200]])
M = cv2.getPerspectiveTransform(pts1,pts2)

while(1):

    frame1 = fvs1.read()

    if np.shape(frame1) == (): # check för empty frame
        continue

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.warpPerspective(frame1,M,(250,200))
    frame1 = cv2.copyMakeBorder(frame1, 3, 3, 3, 3, cv2.BORDER_CONSTANT, 1) # make black frame around image

    #frame1 = _find_edges_laplacian(frame1, 3)
    #frame1 = cv2.adaptiveThreshold(frame1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    frame1 = cv2.adaptiveThreshold(frame1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #frame1 = auto_canny(frame1)

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

    #frame2 = fvs2.read()
    #frame2 = frame2[100:300, 650:900]
    #Scv2.imshow('VIDEO2', frame2)

    cv2.waitKey(1)
