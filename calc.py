#本程序用于计算空间位置
import cv2#4.4.0
import os,sys
import numpy as np
import time
from scipy.optimize import leastsq   

import cv2.aruco as aruco
import glob
import matplotlib.pyplot as plt
from torch_method import *
from cv_method import *
from GrabImage import *

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")#训练设备


a = 0.025#正方体棱长2.5cm
pl = 0.134#笔长13.4cm
d = 0.022#marker边长


flag = 1
r_lr=1e-6
t_lr=1e-8
cali_choice = 'GN'


std_id = 8#标准坐标系标签
top_id = 1#顶端标签
up_ids=[2,3,4,5]#上层标签

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)



def get_camera_mtx(fresh=False):
    '''获取相机矩阵'''


    if fresh:
        # 找棋盘格角点

        cam,stFrameInfo,nPayloadSize = init_cam()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        #棋盘格模板规格
        w = 11   # 12 - 1
        h = 8   # 9  - 1

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        objp = objp*(15)  # 棋盘格边长15mm

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点


        i = 0 

    
        while True:

            ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,1)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame',2592//2, 2048//2)
            cv2.imshow("frame",frame)

            key = cv2.waitKey(1)

            if key == 27:         # 按esc键退出
                print('esc break...')
                stop_cam(cam)
                cv2.destroyAllWindows()
                break

            if key == ord(' '):   # 按空格键保存
                i=i+1 
                filename = f"calibs/{i}.jpg"
                print(f"frame{i} saved")
                cv2.imwrite(filename, frame)


        imgshape=(0,0)
        images = glob.glob('calibs/*.jpg')  #   拍摄的棋盘图片所在目录

        for fname in images:

            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imgshape=gray.shape[::-1]
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
            # 如果找到足够点对，将其存储起来
            if ret == True:

                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)

                # 将角点在图像上显示
                cv2.drawChessboardCorners(img, (w,h), corners, ret)
                cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('findCorners', 2592//2, 2048//2)
                cv2.imshow('findCorners',img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape, None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imgshape, 0, imgshape)


        print("ret:",ret  )

        #flag = int(input("flag:"))
        np.savez(f'camera_mtx{flag}',mtx=mtx,dist=dist,newcameramtx=newcameramtx,roi=roi)

    #flag = int(input("flag:"))
    camera_para = np.load(f'camera_mtx{flag}.npz')
    mtx=camera_para['mtx']
    dist=camera_para['dist']
    newcameramtx = camera_para['newcameramtx']
    roi = camera_para['roi']

    return mtx,dist,newcameramtx,roi





def calibrate_img(img,mtx,dist,newcameramtx,roi):
    '''利用相机矩阵校准图片'''


    # undist = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # x, y, w1, h1 = roi
    # undist = undist[y:y + h1, x:x + w1]



    return img






def get_trans_mtx(cap_para,fresh=False):
    '''获取坐标系转换矩阵'''


    if fresh:

        cam,stFrameInfo,nPayloadSize = init_cam()
        mtx,dist,newcameramtx,roi = get_camera_mtx(0)
        filename = "trans.jpg"

        while True:

            ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,1)
            frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)
            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters =  aruco.DetectorParameters_create()

    

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            # if len(corners)>0:
            #     gray = zmask(gray,corners,cap_para)
            #     gray = Binarization(gray,corners)

            #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.03 , mtx, dist)

            if ids is not None:
                for i in range(rvec.shape[0]):
                    aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners, ids)
            

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 2592//2, 2048//2)
            cv2.imshow("frame",frame)

            key = cv2.waitKey(1)

            if key == ord(' '):   # 按空格键保存
                print("frame saved")
                cv2.imwrite(filename, raw_frame)
                stop_cam(cam)
                cv2.destroyAllWindows()
                break

            if key == 27:         # 按esc键退出
                print('esc break...')
                stop_cam(cam)
                cv2.destroyAllWindows()

                break



    
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        # if len(corners)>0:
        #     gray = zmask(gray,corners,cap_para)
        #     gray = Binarization(gray,corners)

        #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        if  ids is not None:
            #index = ids[0].tolist().index(8)
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.03 , mtx, dist)

            if cali_choice == 'GN':
                rvec , tvec = GN(corners,rvec,tvec,0.03,50)
            else:  
                rvec , tvec = calibrate_poses(rvec,tvec,corners,mtx,0.03,r_lr,t_lr,0.05,200)

            rvec = rvec[0]
            tvec = tvec[0]
            rmtx,jacobin=cv2.Rodrigues(rvec)#求解旋转矩阵 X*=RX+T
            rmtx = np.linalg.inv(rmtx)#R'

            np.savez('trans_mtx.npz',rmtx=rmtx,tvec=tvec)

        else: 
            raise RuntimeError("marker not found")
    
    trans_para = np.load('trans_mtx.npz')
    tran_mtx = trans_para['rmtx']
    trans_tvec = trans_para['tvec']

    return tran_mtx,trans_tvec





def fun(p,x):
    x=x.reshape([-1,3])
    return np.dot(x,p)


def residuals(p, y, x):
    #实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
    #print(y - fun(p,x))
    return y - fun(p,x)


def get_tip_tvec(cap_para,fresh=False):
    '''获取笔尖坐标（最小二乘）'''

    P0 = [[0], [0], [-pl-a]]


    if fresh:
        cam,stFrameInfo,nPayloadSize = init_cam()
        mtx,dist,newcameramtx,roi = get_camera_mtx(0)
        tcount=1
        while True:

            ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,1)
            frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)
            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters =  aruco.DetectorParameters_create()

    

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            # if len(corners)>0:
            #     gray = zmask(gray,corners,cap_para)
            #     gray = Binarization(gray,corners)

            #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)


            if ids is not None:
                for i in range(rvec.shape[0]):
                    aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners, ids)
            

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 2592//2, 2048//2)
            cv2.imshow("frame",frame)

            key = cv2.waitKey(1)

            if key == ord(' '):   # 按空格键保存
                print("frame saved")
                cv2.imwrite(f"tips/tip{tcount}.jpg", raw_frame)
                cv2.imwrite(f"tips1/tip{tcount}.jpg", frame)
                tcount+=1


            if key == 27:         # 按esc键退出
                print('esc break...')
                stop_cam(cam)
                cv2.destroyAllWindows()
                break



    
        images = glob.glob('tips/*.jpg') 
        
        
        Rs=[]
        ts=[]
    
        for filename in images:


            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters =  aruco.DetectorParameters_create()

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            id_list = ids.ravel('F').tolist()
            index = id_list.index(top_id)

            # if len(corners)>0:
            #     gray = zmask(gray,corners,cap_para)
            #     gray = Binarization(gray,corners)

            #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

            #rvec , tvec = GN(corners,rvec,tvec,d,10)
            

            R,_ = cv2.Rodrigues(rvec[index])
            Rs.append(R)
            ts.append(tvec[index])

        tip_tvecs=[]
        Rds=[]
        tds=[]
        for i in range(len(Rs)-1):
            for j in range(i+1,len(Rs)):
                R1=Rs[i]
                R2=Rs[j]
                t1=ts[i]
                t2=ts[j]
                R = (R1-R2)
                t = t2-t1
                Rds.append(R)
                tds.append(t)
        Rds = np.array(Rds)
        tds = np.array(tds)


        #使用最小二乘法
        x=Rds
        y1=tds
        n=np.shape(x)[0]
        #print(np.shape(x))
        sum=0.
        for i in range(n):
            tmp=np.dot(x[i],P0)-y1[i].T
            print(tmp)
            sum=sum+abs(tmp[0][0])+abs(tmp[1][0])+abs(tmp[2][0])
        print(sum)
        result_fit1 = leastsq(residuals,P0,args=(y1.ravel(),x.ravel()))#拟合函数
        tip_tvec = result_fit1[0]
        np.savez('tip_tvec',tip_tvec=tip_tvec)

    tip_tvecz = np.load('tip_tvec.npz')
    tip_tvec = tip_tvecz['tip_tvec']

    return tip_tvec


def get_tip_tvec1(cap_para,fresh=False):
    '''获取笔尖坐标'''

    if fresh:
        cam,stFrameInfo,nPayloadSize = init_cam()
        mtx,dist,newcameramtx,roi = get_camera_mtx(0)
        tcount=1
        while True:

            ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,1)
            frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)
            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters =  aruco.DetectorParameters_create()

    

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            # if len(corners)>0:
            #     gray = zmask(gray,corners,cap_para)
            #     gray = Binarization(gray,corners)

            #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

            if ids is not None:
                for i in range(rvec.shape[0]):
                    aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners, ids)
            

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 2592//2, 2048//2)
            cv2.imshow("frame",frame)

            key = cv2.waitKey(1)

            if key == ord(' '):   # 按空格键保存
                print("frame saved")
                cv2.imwrite(f"tip{tcount}.jpg", raw_frame)
                tcount+=1
                if(tcount>=3):
                    stop_cam(cam)
                    cv2.destroyAllWindows()
                    break



            if key == 27:         # 按esc键退出
                print('esc break...')
                stop_cam(cam)
                cv2.destroyAllWindows()
                break


    
        img = cv2.imread("tip1.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)


        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.03 , mtx, dist)

        rvec , tvec = GN(corners,rvec,tvec,0.03,50)
        

        R,_ = cv2.Rodrigues(rvec[0])
        T=tvec[0]



        img = cv2.imread("tip2.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

        rvec , tvec = GN(corners,rvec,tvec,d,50)
        

        r,_ = cv2.Rodrigues(rvec[0])
        t=tvec[0]


        dt=T-t
        rinv=np.linalg.inv(r)
        tip_tvec = np.dot(rinv,dt.T)

        np.savez('tip_tvec',tip_tvec=tip_tvec)

    tip_tvecz = np.load('tip_tvec.npz')
    tip_tvec = tip_tvecz['tip_tvec']

    return tip_tvec




def calc_trans(r_dict, t_dict ,rvec1, rvec2, tvec1, tvec2, id1, id2=-1):

    R1=np.zeros((3,3),dtype=np.float64)
    R2=np.zeros((3,3),dtype=np.float64)
    cv2.Rodrigues(rvec1,R1)
    cv2.Rodrigues(rvec2,R2)
    R=np.dot(np.linalg.inv(R1),R2)
    T=np.dot(np.linalg.inv(R1),(tvec2-tvec1).T)
    if id2==-1:
        r_dict[id1]=R
        t_dict[id1]=T
    else:
        r_dict[id1]=np.dot(R, r_dict[id2])
        t_dict[id1]=np.dot(np.linalg.inv(R1),(tvec2-tvec1).T+np.dot(R2,t_dict[id2]))
    return r_dict,t_dict




def trans_M(r_dict, t_dict):

    RM=np.zeros((13,3,3),dtype=np.float64)
    TM=np.zeros((13,1,3),dtype=np.float64)

    for key in r_dict:
        RM[key]=r_dict[key]
    for key in t_dict:
        TM[key]=t_dict[key].T

    RM[top_id] = np.eye(3)
    return RM,TM


def get_marker_pics(cap_para):
    '''拍摄标记照片用于标定'''

    cam,stFrameInfo,nPayloadSize = init_cam()

    mtx,dist,newcameramtx,roi = get_camera_mtx(0)

    while True:

        ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,1)
        frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)
        raw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        # if len(corners)>0:
        #     gray = zmask(gray,corners,cap_para)
        #     gray = Binarization(gray,corners)
        
        #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

        if ids is not None:
            for i in range(rvec.shape[0]):
                aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners, ids)

            ###### DRAW ID #####
            cv2.putText(frame, "Id: " + str(ids.tolist()), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 2592//2, 2048//2)
        cv2.imshow("frame",frame)

        key = cv2.waitKey(1)

        if key == 27:         # 按esc键退出
            print('esc break...')
            stop_cam(cam)
            cv2.destroyAllWindows()

            break

        if key == ord(' '):   # 按空格键保存

            current_id = int(input("input id:"))
            filename = f"marker_calibs/marker{current_id}.jpg"
            cv2.imwrite(filename, raw_frame)
            filename = f"marker_calibs1/marker{current_id}.jpg"
            cv2.imwrite(filename, frame)

            print(f"marker{current_id} saved")


def get_marker_mtx(cap_para,fresh=False):
    '''标定标记坐标系转换矩阵'''


    r_dict={}
    t_dict={}
    
    if fresh:

        mtx,dist,newcameramtx,roi = get_camera_mtx(0)

        images = glob.glob('marker_calibs/*.jpg')  #   拍摄的棋盘图片所在目录

        while len(images) != 4:
            get_marker_pics(cap_para)
            images = glob.glob('marker_calibs/*.jpg')  #   拍摄的棋盘图片所在目录

       
        for current_id in up_ids:

            filename = f"marker_calibs/marker{current_id}.jpg"


            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters =  aruco.DetectorParameters_create()

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            # if len(corners)>0:
            #     gray = zmask(gray,corners,cap_para)
            #     gray = Binarization(gray,corners)

            #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

            rvec , tvec = GN(corners,rvec,tvec,d,50)
            

            assert rvec.shape[0] == 2
            
            id_list = ids.ravel('F').tolist()
            index = id_list.index(current_id)

            rvec1 = rvec[index]
            rvec2 = rvec[1-index]
            tvec1 = tvec[index]
            tvec2 = tvec[1-index]

            r_dict,t_dict = calc_trans(r_dict,t_dict,rvec1,rvec2,tvec1,tvec2,current_id)


        

        r_mtx , t_mtx = trans_M(r_dict,t_dict)
        np.savez('marker_mtx',r_mtx = r_mtx,t_mtx = t_mtx)

    marker_para = np.load('marker_mtx.npz')
    r_mtx = marker_para['r_mtx']
    t_mtx = marker_para['t_mtx']

    return r_mtx , t_mtx





##############################轨迹检测###############################


def get_half_nearest(rlist,tlist,id_list,tip_tvec,t_threshold=0.015,r_threshold=0.15):
    '''从n个点中找出n/2近邻'''

    n = len(id_list)
    index1=0
    index2=n//2
    if 1 in id_list:
        index1 = id_list.index(1)
        index2 = index1 + n//2

    tlist1 = [tlist[index1]]
    rlist1 = [rlist[index1]]
    tlist2 = [tlist[index2]]
    rlist2 = [rlist[index2]]#初始化聚类中心

    for i in range(n):
        if i==index1 or i==index2:continue

        tdist1 = np.linalg.norm(tlist[i]-tlist1[0])
        tdist2 = np.linalg.norm(tlist[i]-tlist2[0])
        rdist1 = np.linalg.norm(rlist[i]-rlist1[0])
        rdist2 = np.linalg.norm(rlist[i]-rlist2[0])



        if tdist1<t_threshold and rdist1<r_threshold:
            rlist1.append(rlist[i])
            tlist1.append(tlist[i])
        elif tdist2<t_threshold and rdist2<r_threshold:
            rlist2.append(rlist[i])
            tlist2.append(tlist[i])

    
    if len(rlist1)>=len(rlist2):
        correct_rlist = np.array(rlist1)
        correct_tlist = np.array(tlist1)
    else:
        correct_rlist = np.array(rlist2)
        correct_tlist = np.array(tlist2)

    
    r_mean = np.mean(correct_rlist,axis=0)
    t_mean = np.mean(correct_tlist,axis=0)


    r_std = np.std(correct_rlist,axis=0)
    t_std = np.std(correct_tlist,axis=0)

    r_std = np.linalg.norm(r_std)**0.5
    t_std = np.linalg.norm(t_std)**0.5


    z = tip_tvec
    R,jacobin = cv2.Rodrigues(r_mean)
    pos = np.dot(R,z).T+t_mean#计算笔尖坐标

    # if(len(correct_rlist)==1):
    #     pos = None

    return r_mean,t_mean,r_std,t_std ,pos



def calc_pose(rvec , tvec, ids, r_mtx , t_mtx,tip_tvec):
    '''根据marker求解笔pose与笔尖坐标pos'''

    id_list = ids.ravel('F').tolist()
    id_list = id_list + id_list

    nums = rvec.shape[0]

    pose_r_list=[]
    pose_t_list=[]

    for i in range(nums):

        #get pose_r
        current_id = id_list[i]
        R,jacobin = cv2.Rodrigues(rvec[i])
        Q = r_mtx [current_id]
        pose_r_mtx = np.dot(R,Q)
        pose_r,jacobin = cv2.Rodrigues(pose_r_mtx)

        pose_r_list.append(pose_r.T)#pose_r 1*3

        #get pose_t
        global_t = np.dot(R,t_mtx[current_id].T)
        global_t = global_t + tvec[i].T
        pose_t = global_t.T

        pose_t_list.append(pose_t)#pose_t 1*3

    
    r_mean,t_mean,r_std,t_std,pos = get_half_nearest(pose_r_list,pose_t_list,id_list,tip_tvec)


    return r_mean,t_mean,r_std,t_std,pos






def rvec_calibrate(rvecs,tvecs,corners,mtx,d):

    '''纠正错误z轴方向'''

    #fatal error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n=np.shape(rvecs)[0]

    correct_rvecs = []
    other_rvecs = []

    for i in range(n):

    

        rvec1 = rvecs[i].copy()
        tvec = tvecs[i].copy()
        corner = corners[i].copy()

        R , jacobin = cv2.Rodrigues(rvec1)
        temp = R[:,0].copy()
        R[:,0]=R[:,1]
        R[:,1]=temp
        R[:,2]=-R[:,2]

        rvec2 , jacobin = cv2.Rodrigues(R)
        rvec2 = rvec2.T


        # rvec1 = torch.from_numpy(rvec1).to(device)
        # rvec2 = torch.from_numpy(rvec2).to(device)
        # tvec = torch.from_numpy(tvec).to(device)

        # loss1 = calc_loss(rvec1,tvec,corner,mtx,d)
        # loss2 = calc_loss(rvec2,tvec,corner,mtx,d)
        
        
        
        # rvec1 = rvec1.cpu().detach().numpy()
        # rvec2 = rvec2.cpu().detach().numpy()
        correct_rvecs.append(rvec1)
        other_rvecs.append(rvec2)
        #correct_rvecs.append(rvec1 if loss1<loss2 else rvec2)

    correct_rvecs = np.array(correct_rvecs)
    other_rvecs = np.array(other_rvecs)
        
    return correct_rvecs, other_rvecs



def check_av(rvecs , tvecs,corners,ids):
    '''检查检测结果是否合法'''

    z = np.array([[0,0,1]]).T

    i=0

    while i < np.shape(rvecs)[0]:
        
        R,jacobin = cv2.Rodrigues(rvecs[i])
        Z = np.dot(R,z)
        Z = Z.T

        if ids[i][0] not in  (up_ids+[top_id]):
            print("error")
            rvecs = np.delete(rvecs,i,axis=0)
            tvecs = np.delete(tvecs,i,axis=0)
            ids = np.delete(ids,i,axis=0)
            corners= np.delete(corners,i,axis=0)

        elif Z[0][2]>0:
            print("error")
            rvecs = np.delete(rvecs,i,axis=0)
            tvecs = np.delete(tvecs,i,axis=0)
            ids = np.delete(ids,i,axis=0)
            corners= np.delete(corners,i,axis=0)

        else : i= i + 1
    

    if np.shape(ids)[0]==0: ids = None
    

    return rvecs,tvecs,corners,ids






def Target_tracking(Frame_slice, Frame,rgb,old_gray,prev_corners,old_ids):
    
    if rgb: 
        Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        gray=Frame[Frame_slice[0]:Frame_slice[1],Frame_slice[2]:Frame_slice[3]]

        
        #ret,gray=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    else:
        frame = Frame[Frame_slice[0]:Frame_slice[1],Frame_slice[2]:Frame_slice[3]]
        frame = frame.astype(np.uint8)
        gray = frame  
        #ret,gray=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)


    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters =  aruco.DetectorParameters_create()


   
    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    if Frame_slice[0]==0 and Frame_slice[1]==2048 and Frame_slice[2]==0 and Frame_slice[3]==2592:
        temp = cv2.resize(gray,(2592//2,2048//2))#降采样提高检测效率
        corners, ids, rejectedImgPoints = aruco.detectMarkers(temp,aruco_dict,parameters=parameters)
        corners = np.array(corners)
        corners=corners*2

    
    else:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        corners = np.array(corners)



    frame_slice=np.array([0,2048,0,2592])

    if ids is None:
        # frame_slice=np.array([0,2048,0,2592])
        # Frame_slice=np.array([0,2048,0,2592])

        # if old_gray.shape[0]!=0 and prev_corners.shape[0]!=0 and old_ids.shape[0]!=0:

            
        #     corners_n=prev_corners.shape[0]
        #     prev_corners=prev_corners.reshape(corners_n*4,1,2)
        #     tracked_corners = track_corners(prev_corners,old_gray,Frame)
        #     tracked_corners =tracked_corners.reshape(corners_n,1,4,2)


        #     old_gray = Frame
        #     prev_corners = tracked_corners


        #     corners2=np.array(tracked_corners).astype(int)
        #     n=np.shape(corners2)[0]
        #     x1=corners2[0:n,0:1,0:4,0:1].min()+Frame_slice[2]
        #     x2=corners2[0:n,0:1,0:4,0:1].max()+Frame_slice[2]
        #     y1=corners2[0:n,0:1,0:4,1:2].min()+Frame_slice[0]
        #     y2=corners2[0:n,0:1,0:4,1:2].max()+Frame_slice[0]
        #     frame_slice[2]=max(0,x1-100)
        #     frame_slice[0]=max(0,y1-100)
        #     frame_slice[3]=min(2592,x2+100)
        #     frame_slice[1]=min(2048,y2+100)
        #     tracked_corners=np.array(tracked_corners)
        #     tracked_corners[0:n,0:1,0:4,0:1]+=Frame_slice[2]
        #     tracked_corners[0:n,0:1,0:4,1:2]+=Frame_slice[0]

        #     ids=old_ids
            

        #     return tracked_corners,ids,frame_slice,old_gray,prev_corners,old_ids

        pass

    else:

        # print(corners)
        corners2=np.array(corners).astype(int)
        n=np.shape(corners2)[0]
        x1=corners2[0:n,0:1,0:4,0:1].min()+Frame_slice[2]
        x2=corners2[0:n,0:1,0:4,0:1].max()+Frame_slice[2]
        y1=corners2[0:n,0:1,0:4,1:2].min()+Frame_slice[0]
        y2=corners2[0:n,0:1,0:4,1:2].max()+Frame_slice[0]
        frame_slice[2]=max(0,x1-200)
        frame_slice[0]=max(0,y1-200)
        frame_slice[3]=min(2592,x2+200)
        frame_slice[1]=min(2048,y2+200)
        corners=np.array(corners)
        corners[0:n,0:1,0:4,0:1]+=Frame_slice[2]
        corners[0:n,0:1,0:4,1:2]+=Frame_slice[0]

        old_ids = ids
        old_gray = Frame
        prev_corners = corners
    
    return corners,ids,frame_slice,old_gray,prev_corners,old_ids