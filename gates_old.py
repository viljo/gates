import cv2
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
import imutils
import math
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

fvs1 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/101/").start()
fvs2 = FileVideoStream("rtsp://guest:0000@home.viljo.se:554/Streaming/Channels/201/").start()

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def frame_treatment(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(3,3),0)
    frame = auto_canny(frame)
    return frame

def line_detect(frame):
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = cv2.GaussianBlur(f,(3,3),0)
    f = auto_canny(f)
    minLineLength = 15
    maxLineGap = 10
    lines = cv2.HoughLinesP(f,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    return frame

def _find_edges_laplacian(image, edge_multiplier):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def poly_detect(frame):
    #f = _find_edges_laplacian(frame, 3)
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #f = cv2.Laplacian(f, cv2.CV_16S, ksize=1)
    #f = cv2.convertScaleAbs(f)
    #f = auto_canny(f)
    kernel = np.ones((1,1), np.uint8)
    #f = cv2.erode(f, kernel, iterations=1) 
    f = cv2.dilate(f, kernel, iterations=1) 
    #f = cv2.GaussianBlur(f,(3,3),0)
    
    #ret, f = cv2.threshold(f, 240, 255, cv2.THRESH_BINARY)
    f=cv2.bitwise_not(f)
    contours, hierarchy= cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 1) 

    for cnt in contours:
        if (cv2.contourArea(cnt) > 50):
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            if len(approx) >= 4:
                cv2.drawContours(frame, [approx], 0, (0,255,0), 1)
    cv2.imshow('VIDEO1lr', cv2.resize(f,None,fx=3,fy=3))
    return frame

def line_detect_p(frame):
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #f = auto_canny(f)
    f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)  
    minLineLength = 200
    maxLineGap = 10
    lines = cv2.HoughLinesP(f,1,np.pi/180,100,minLineLength,maxLineGap)

    a,b,c = lines.shape
    for i in range(a):
        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('VIDEO1lr', cv2.resize(f,None,fx=3,fy=3))
    return frame

def line_detect(frame):
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #f = auto_canny(f)
    f = cv2.adaptiveThreshold(f,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)  
    
    lines = cv2.HoughLines(f,1,np.pi/180,10)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imshow('VIDEO1lr', cv2.resize(f,None,fx=3,fy=3))
    return frame

x1=0
x2=0
y1=0
y2=0
while(1):

    frame1 = fvs1.read()
    #frame1 = cv2.resize((frame1),None,fx=0.25,fy=0.25)
    frame1 = frame1[50:320, 240:540]
    #frame1 = frame1[1:1, 240:540]
    #frame1 = imutils.rotate(frame1, angle=-18)

    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    #pts = np.array([(860-x1, 495-y1), (1400-x2, 343-y2), (1431-x2, 676-y2), (911-x1, 842-y1)], dtype = "float32")
    #pts = np.array([(911, 842), (1431, 676), (1400, 343), (860, 495)], dtype = "float32")
    #frame1 = four_point_transform(frame1, pts)

    rows,cols,ch = frame1.shape
    pts1 = np.float32([(185, 693),(708, 519),(708, 519),(136, 344)])
    pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    frame1 = cv2.warpPerspective(frame1,M,(500,500))


    cv2.imshow('VIDEO1', frame1)
    cv2.imshow('VIDEO1t', frame_treatment(frame1))
 

    frame2 = fvs2.read()
    frame2 = frame2[100:300, 650:900]
    cv2.imshow('VIDEO2', frame2)
    cv2.imshow('VIDEO2t', frame_treatment(frame2))

    cv2.imshow('VIDEO1l', cv2.resize((frame1),None,fx=3,fy=3))

    cv2.waitKey(1)
