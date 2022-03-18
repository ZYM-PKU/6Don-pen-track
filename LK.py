import numpy as np
import cv2
import sys
from calc import *
import cv2.aruco as aruco

#cap = cv2.VideoCapture("C:\\Users\\刘旭\\Pictures\\Camera Roll\\video2.mp4")
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)#选取好的特征点，返回特征点列表
mask = np.zeros_like(old_frame)

ret,frame = cap.read()
time.sleep(1)
ret,frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
corners=np.array(corners)
corners_n=corners.shape[0]
p0=corners.reshape(corners_n*4,1,2)


while(1):
    ret,frame = cap.read()
    if frame is None: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)#计算新的一副图像中相应的特征点额位置
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #ravel()函数用于降维并且不产生副本
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord(' '):
        mask = np.zeros_like(old_frame)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()