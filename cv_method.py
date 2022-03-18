import cv2#4.4.0
import os,sys,glob
import numpy as np
import time
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import pickle
import math

#指定工作目录
PATH = os.path.dirname(__file__)
os.chdir(PATH)


top_id=1
up_ids=[2,3,4,5]#上层标签




def track_corners(prev_corners,old_gray,frame_gray):
    lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    current_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_corners, None, **lk_params)
    return current_corners


def zmask(gray,corners,cap_para) :
    '''针对marker做掩膜运算'''
    n = len(corners)
    

    mask_size = (cap_para['h'],cap_para['w'])
    mask = np.zeros(mask_size, dtype = np.uint8)
    mask = 255-mask

    for i in range(n):
        corner = corners[i].astype(int)
        corner_w = np.max(corner[0,:,0])-np.min(corner[0,:,0])
        corner_h = np.max(corner[0,:,1])-np.min(corner[0,:,1])
        corner_size = (corner_w + corner_h)/2
        padding = int(corner_size/10)+1
        cv2.polylines(mask, corner, True, 0 , padding)
        cv2.fillPoly(mask, corner, 0)
    
    gray = np.bitwise_or(gray,mask)


    return gray


def Binarization(gray,corners):
    #thresh1=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
    corners=np.array(corners).astype(int)
    n=np.shape(corners)[0]
    x1=corners[0:n,0:1,0:4,0:1].min()
    x2=corners[0:n,0:1,0:4,0:1].max()
    y1=corners[0:n,0:1,0:4,1:2].min()
    y2=corners[0:n,0:1,0:4,1:2].max()
    rect=gray[y1:y2,x1:x2]
    threshold=np.mean(rect)*(0.9)
    ret,thresh1 = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    return thresh1





def Binarization2(gray,corners):
    #thresh1=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
    corners=np.array(corners).astype(int)
    n=np.shape(corners)[0]
    rect_corners=np.zeros((n,4),dtype=int)
    contrast=np.zeros((n,1))
    mean=np.zeros((n,1))
    max_contrast=0
    max_contrast_id=0
    max_mean=0
    max_mean_id=0
    for i in range(n):
        rect_corners[i][0]=corners[i:i+1,0:1,0:4,0:1].min()
        rect_corners[i][1]=corners[i:i+1,0:1,0:4,0:1].max()
        rect_corners[i][2]=corners[i:i+1,0:1,0:4,1:2].min()
        rect_corners[i][3]=corners[i:i+1,0:1,0:4,1:2].max()

        rect=gray[rect_corners[i][2]:rect_corners[i][3],rect_corners[i][0]:rect_corners[i][1]]
        contrast[i][0]=rect.std()
        mean[i][0]=np.mean(rect)
        if contrast[i][0]>max_contrast:
            max_contrast=contrast[i][0]
            max_contrast_id=i
        if mean[i][0]>max_mean:
            max_mean=mean[i][0]
            max_mean_id=i
    for i in range(n):
        rect_area = gray[rect_corners[i][2]:rect_corners[i][3],rect_corners[i][0]:rect_corners[i][1]]
        bonus = max_mean-mean[i][0]
        rect_area = rect_area+bonus

        new_rect_area = np.bitwise_and(rect_area.astype(int),255)

        area1 = rect_area - new_rect_area
        area2 = 255 - area1

        new_rect_area = new_rect_area + area2    

        rect_area = new_rect_area%255   

        

    return gray


def rotationMatrixToEulerAngles(R) :

    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



def estimatePoseMarkers_Z(corners, d , mtx, dist ,ids , t_mtx):
    '''ZYM重写的estimatePoseMarker函数'''

    id_list = ids.ravel('F').tolist()


    rvecs = []
    tvecs = []

    for i in range(len(id_list)):
        obj_point = np.array([
        [-d/2,d/2,0],
        [d/2,d/2,0],
        [d/2,-d/2,0],
        [-d/2,-d/2,0]
        ])
        current_id = id_list[i]
        corner = corners[i][0]
        
        if 1 in id_list:
            obj_point = np.append(obj_point,t_mtx[current_id],axis=0)
            top_corner = np.mean(corners[id_list.index(1)][0],axis=0).reshape(1,2)
            corner = np.append(corner,top_corner,axis=0)
            
        _,c_rvec,c_tvec = cv2.solvePnP(obj_point,corner,mtx,dist,flags=cv2.SOLVEPNP_UPNP,useExtrinsicGuess=True)
        R , jacobin= cv2.Rodrigues(c_rvec)
        R = np.linalg.inv(R)
        rvec , jacobin = cv2.Rodrigues(R)
        rvecs.append(rvec.T)
        tvecs.append(c_tvec.T)
    

    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return rvecs , tvecs



                
    

if __name__ == "__main__":
    cap_para={'choice':0,'w':2560,'h':1440,'fps':30}
    fnames = glob.glob("marker_calibs_up/*.jpg")
    count=1
    for f in fnames:
        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters_create()

        
        #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        if len(corners)>0:
            gray = zmask(gray,corners,cap_para)
            #gray = Binarization(gray,corners)

        
        cv2.imwrite(f"grays/{count}.jpg",gray)
        count = count+1