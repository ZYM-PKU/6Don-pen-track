#本程序用于检测aruco标记
import cv2#4.4.0
import os,sys
import numpy as np
import time

from numpy.core.records import record
import cv2.aruco as aruco
import matplotlib.pyplot as plt


from calc import *
from torch_method import *
from cv_method import *
from GrabImage import *


#test my draw
import DobotDllType as dType
import math
import threading
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}


#指定工作目录
PATH = os.path.dirname(__file__)
os.chdir(PATH)



#基本参数设定

a = 0.0129#正12面体棱长#0.0129m
r = a*0.5*((2.5+1.1*(5**0.5))**0.5)#内切球半径#0.0143643611m
pl = 0.134 + 2*r #笔长13.4cm

cap_para={'choice':0,'w':2592,'h':2048,'fps':40}
rgb=0#是否采用彩色模式

r_lr=1e-6
t_lr=1e-8
min_loss=0.1
max_iters=5
d = 0.0221
mindis = 0.001#最小笔尖距




cover = 0.3#画布范围（单位：m）
draw_board = np.ones((int(cover*3000),int(cover*3000),3),np.dtype('uint8'))*255#画布
window_size = 8
filter_window= np.ones(window_size,dtype=np.float)/window_size #滑动窗口
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)




#标定与校准
mtx,dist,newcameramtx,roi = get_camera_mtx(0)
print("mtx:\n",mtx)
print("dist:\n",dist)
#tvec_mtx  = get_tvec_mtx(cap_para,0)
r_mtx , t_mtx = get_marker_mtx(cap_para,0)
tip_tvec = get_tip_tvec(cap_para,0).reshape((3,1))
tran_mtx, trans_tvec= get_trans_mtx(cap_para,1)
# tran_mtx, trans_tvec= get_trans_mtx1(cap_para,tip_tvec,0)


bot_enable = 0#是否开启机械臂


def get_video():

    cam,stFrameInfo,nPayloadSize = init_cam()

    i=int(input("video id:"))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output{i}.avi',fourcc, cap_para['fps'], (cap_para['w'],cap_para['h']),isColor=False)
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    while True:

        ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,rgb)
        out.write(frame)
        temp = cv2.resize(frame,(2592//2,2048//2))
        cv2.imshow('frame', temp)
        key = cv2.waitKey(1)

        if key == 27:         # 按esc键退出
            print('esc break...')
            stop_cam(cam)
            out.release()
            cv2.destroyAllWindows()
            break






def online_run(draw=1):
    global draw_board

    if bot_enable:
        #Clean Command Queued
        dType.SetQueuedCmdClear(api)
        
        #设置运动参数
        #Async Motion Params Setting
        #set home
        dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
        
        dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
        dType.SetPTPCoordinateParams(api,200,200,200,200,isQueued = 1)#test queue
        dType.SetPTPJumpParams(api,10,200,isQueued = 1)
        dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
        #归零??
        #dType.SetHOMECmd(api, temp = 0, isQueued = 1)
        
        print("enable rail is ",dType.GetDeviceWithL(api))

        pos = dType.GetPose(api)
        print(pos)
        X=pos[0]
        Y=pos[1]
        Z=pos[2]
        rHead = pos[3]
        
        L = dType.GetPoseL(api)[0]
        print("current rail pos ",L)


        flag=True

    cam,stFrameInfo,nPayloadSize = init_cam()

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

    if draw:
        cv2.namedWindow('board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('board', int(cover*3000),int(cover*3000))
    
    

    #3d坐标列
    pos_3d_x=np.array([])
    pos_3d_y=np.array([])
    pos_3d_z=np.array([])


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

    last_2d_x = None
    last_2d_y = None#实时绘制数据保存

    tic = time.time()-1

    corners, ids=0,0
    frame_slice = np.array([0,2048,0,2592])

    old_gray=np.array([])
    prev_corners=np.array([])
    old_ids = np.array([])

    pre_rs=[]
    pre_ts=[]

    scaler = 0.6#放缩系数


    while True:



        ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,rgb)

        

        toc = time.time()
        time_dure = toc-tic
        current_fps = 1/time_dure 
        tic = time.time()

        cv2.putText(frame, f"fps:  {current_fps}", (2592-400,64), font, 1, (0,0,255),2,cv2.LINE_AA)

        #frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)

        corners,ids,frame_slice,old_gray,prev_corners,old_ids = Target_tracking(frame_slice,frame,rgb,old_gray,prev_corners,old_ids)

        corners=np.array(corners)


        rvec1 = 0
        tvec1 = 0
        

        #位姿检测
        if ids is not None:

            rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

            rvec1,tvec1,corners,ids = check_av(rvec1,tvec1,corners,ids)#检测坏点

            
        #计算笔尖坐标
        if ids is not None:

            #rvec1 , tvec1 = GN(corners,rvec1,tvec1,d,3)#高斯牛顿迭代
            rvec1 , rvec2 = rvec_calibrate(rvec1,tvec1,corners,mtx,d)#双生坐标系


            for i in range(rvec1.shape[0]):
                aruco.drawAxis(frame, mtx, dist, rvec1[i, :, :], tvec1[i, :, :], 0.03)
                #aruco.drawAxis(frame, mtx, dist, rvec2[i, :, :], tvec1[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners, ids)


            ###### DRAW ID #####
            cv2.putText(frame, "Id: " + str(ids.tolist()), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            cv2.rectangle(frame, (frame_slice[2],frame_slice[0]),(frame_slice[3],frame_slice[1]),(0,0,255),2)

        
        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        


        if ids is not None:
            rvec = np.append(rvec1,rvec2,axis=0)
            tvec = np.append(tvec1,tvec1,axis=0)

            pre_r,pre_t,r_std,t_std ,pos = calc_pose(rvec , tvec, ids, r_mtx , t_mtx,tip_tvec)#计算笔6Don位姿

            if pos is not None:

                pre_rs.append(pre_r)
                pre_ts.append(pre_t)
                pos_2d=np.dot(mtx,pos.reshape(3,1))
                cal_x=pos_2d[0][0]/pos_2d[2][0]
                cal_y=pos_2d[1][0]/pos_2d[2][0]
                cv2.circle(frame,(int(cal_x),int(cal_y)),2,(0, 0, 255))
        

                pos = np.dot(tran_mtx, (pos-trans_tvec).reshape(3,1)).T[0]#转换到桌面坐标系
                print(pos)
                

                pos_3d_x=np.append(pos_3d_x,pos[0])
                pos_3d_y=np.append(pos_3d_y,pos[1])
                pos_3d_z=np.append(pos_3d_z,pos[2])


                if np.shape(pos_3d_x)[0]>=window_size:
                    new_3d_x = np.sum(pos_3d_x[-window_size:])/window_size
                    new_3d_y = np.sum(pos_3d_y[-window_size:])/window_size
                    new_3d_z = np.sum(pos_3d_z[-window_size:])/window_size

                    if new_3d_z<mindis:
                        new_2d_x = new_3d_x
                        new_2d_y = new_3d_y


                        if bot_enable:
                            x = new_2d_x*1000*scaler
                            y = new_2d_y*1000*scaler

                            #限位判定
                            if y>70:
                                y-=70
                                L-=70
                            if y<-70:
                                y+=70
                                L+=70

                            if flag:
                                lastIndex = dType.SetPTPWithLCmd(api,0,X-y,Y+x,Z,rHead,L,isQueued=1)[0]
                                flag=False
                            else:
                                lastIndex = dType.SetPTPWithLCmd(api,2,X-y,Y+x,Z,rHead,L,isQueued=1)[0]

                    else:
                        new_2d_x = None
                        new_2d_y = None

                        if bot_enable:
                            flag=True


                else:
                    new_2d_x = None
                    new_2d_y = None
                
                if new_2d_x != None and last_2d_x != None and draw:
                    cv2.line(draw_board, 
                    (int(last_2d_x*3000+cover*1500),
                    int(cover*3000)-int(last_2d_y*3000+cover*1500)),
                    (int(new_2d_x*3000+cover*1500),
                    int(cover*3000)-int(new_2d_y*3000+cover*1500)), (255, 0, 0) , 2) 
                    
                last_2d_x = new_2d_x
                last_2d_y = new_2d_y
            
            else :
                print(0)
    


        # 显示结果  
        temp = cv2.resize(frame,(2592//2,2048//2))
        cv2.imshow("frame",temp)
        if draw:
            cv2.imshow("board",draw_board)
        key = cv2.waitKey(1)



        # 按esc键退出
        if key == 27:   

            num = int(input("record num:"))
            np.savez(f'3d_pos{num}.npz',pos_3d_x=pos_3d_x,pos_3d_y=pos_3d_y,pos_3d_z=pos_3d_z)
            stop_cam(cam)
            cv2.destroyAllWindows()

            break


        if key == ord('p'):
            while 1:
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break


        if key == ord(' '):

            draw_board = np.ones((int(cover*3000),int(cover*3000),3),np.dtype('uint8'))*255



def offline_run(draw=True):
    global draw_board

    if bot_enable:
        #Clean Command Queued
        dType.SetQueuedCmdClear(api)
        
        #设置运动参数
        #Async Motion Params Setting
        #set home
        dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
        
        dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
        dType.SetPTPCoordinateParams(api,200,200,200,200,isQueued = 1)#test queue
        dType.SetPTPJumpParams(api,10,200,isQueued = 1)
        dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
        #归零??
        #dType.SetHOMECmd(api, temp = 0, isQueued = 1)
        
        print("enable rail is ",dType.GetDeviceWithL(api))

        pos = dType.GetPose(api)
        print(pos)
        X=pos[0]
        Y=pos[1]
        Z=pos[2]
        rHead = pos[3]
        
        L = dType.GetPoseL(api)[0]
        print("current rail pos ",L)


        flag=True

    i=int(input("video id:"))

    vcap = cv2.VideoCapture(f'output{i}.avi') 
    
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

    if draw:
        cv2.namedWindow('board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('board', int(cover*3000),int(cover*3000))
    
    

    #3d坐标列
    pos_3d_x=np.array([])
    pos_3d_y=np.array([])
    pos_3d_z=np.array([])


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

    last_2d_x = None
    last_2d_y = None#实时绘制数据保存

    tic = time.time()-1

    corners, ids=0,0
    frame_slice = np.array([0,2048,0,2592])

    old_gray=np.array([])
    prev_corners=np.array([])
    old_ids = np.array([])

    pre_rs=[]
    pre_ts=[]

    scaler = 0.6#放缩系数


    while True:

        try:

            ret, frame = vcap.read()
            h, w = frame.shape[:2]
        
        except:

            num = int(input("record num:"))
            np.savez(f'3d_pos{num}.npz',pos_3d_x=pos_3d_x,pos_3d_y=pos_3d_y,pos_3d_z=pos_3d_z)
            vcap.release()
            cv2.destroyAllWindows()

            break


        toc = time.time()
        time_dure = toc-tic
        current_fps = 1/time_dure 
        tic = time.time()

        cv2.putText(frame, f"fps:  {current_fps}", (2592-400,64), font, 1, (0,0,255),2,cv2.LINE_AA)

        #frame = calibrate_img(frame,mtx,dist,newcameramtx,roi)

        corners,ids,frame_slice,old_gray,prev_corners,old_ids = Target_tracking(frame_slice,frame,rgb,old_gray,prev_corners,old_ids)
        #corners: array(n,1,4,2)
        corners=np.array(corners)


        rvec1 = 0
        tvec1 = 0
        


        #位姿检测
        if ids is not None:

            rvec1, tvec1, _ = aruco.estimatePoseSingleMarkers(corners, d , mtx, dist)

            rvec1,tvec1,corners,ids = check_av(rvec1,tvec1,corners,ids)#检测坏点
            # print("rvec\n",rvec1)
            # print("tvec\n",tvec1)
            # print(corners)
            
        #计算笔尖坐标
        if ids is not None:

            #rvec1 , tvec1 = GN(corners,rvec1,tvec1,d,3)#高斯牛顿迭代
            #rvec1 , tvec1 = calibrate_poses(rvec1,tvec1,corners,mtx,d,r_lr,t_lr,min_loss,max_iters)#梯度下降
            rvec1 , rvec2 = rvec_calibrate(rvec1,tvec1,corners,mtx,d)#双生坐标系


            for i in range(rvec1.shape[0]):
                aruco.drawAxis(frame, mtx, dist, rvec1[i, :, :], tvec1[i, :, :], 0.03)
                #aruco.drawAxis(frame, mtx, dist, rvec2[i, :, :], tvec1[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners, ids)


            ###### DRAW ID #####
            cv2.putText(frame, "Id: " + str(ids.tolist()), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            cv2.rectangle(frame, (frame_slice[2],frame_slice[0]),(frame_slice[3],frame_slice[1]),(0,0,255),2)

        
        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        


        if ids is not None:
            rvec = np.append(rvec1,rvec2,axis=0)
            tvec = np.append(tvec1,tvec1,axis=0)

            pre_r,pre_t,r_std,t_std ,pos = calc_pose(rvec , tvec, ids, r_mtx , t_mtx,tip_tvec)#计算笔6Don位姿
            if pos is not None:
                pre_rs.append(pre_r)
                pre_ts.append(pre_t)


        
            if pos is not None:

                pos = np.dot(tran_mtx, (pos-trans_tvec).reshape(3,1)).T[0]#转换到桌面坐标系
                print(pos)
                

                pos_3d_x=np.append(pos_3d_x,pos[0])
                pos_3d_y=np.append(pos_3d_y,pos[1])
                pos_3d_z=np.append(pos_3d_z,pos[2])


                if np.shape(pos_3d_x)[0]>=window_size:
                    new_3d_x = np.sum(pos_3d_x[-window_size:])/window_size
                    new_3d_y = np.sum(pos_3d_y[-window_size:])/window_size
                    new_3d_z = np.sum(pos_3d_z[-window_size:])/window_size

                    if new_3d_z<mindis:
                        new_2d_x = new_3d_x
                        new_2d_y = new_3d_y
                        
                        if bot_enable:
                            x = new_2d_x*1000*scaler
                            y = new_2d_y*1000*scaler

                            #限位判定
                            if y>70:
                                y-=70
                                L-=70
                            if y<-70:
                                y+=70
                                L+=70

                            if flag:
                                lastIndex = dType.SetPTPWithLCmd(api,0,X-y,Y+x,Z,rHead,L,isQueued=1)[0]
                                flag=False
                            else:
                                lastIndex = dType.SetPTPWithLCmd(api,2,X-y,Y+x,Z,rHead,L,isQueued=1)[0]

                    else:
                        new_2d_x = None
                        new_2d_y = None
                        if bot_enable:
                            flag=True


                else:
                    new_2d_x = None
                    new_2d_y = None
                
                if new_2d_x != None and last_2d_x != None and draw:
                    cv2.line(draw_board, 
                    (int(last_2d_x*3000+cover*1500),
                    int(cover*3000)-int(last_2d_y*3000+cover*1500)),
                    (int(new_2d_x*3000+cover*1500),
                    int(cover*3000)-int(new_2d_y*3000+cover*1500)), (255, 0, 0) , 2) 
                    
                last_2d_x = new_2d_x
                last_2d_y = new_2d_y
            
            else :
                print(0)
    


        # 显示结果  
        temp = cv2.resize(frame,(2592//2,2048//2))
        cv2.imshow("frame",temp)
        if draw:
            cv2.imshow("board",draw_board)
        key = cv2.waitKey(1)



        # 按esc键退出
        if key == 27:   

            num = int(input("record num:"))
            np.savez(f'3d_pos{num}.npz',pos_3d_x=pos_3d_x,pos_3d_y=pos_3d_y,pos_3d_z=pos_3d_z)
            stop_cam(cam)
            cv2.destroyAllWindows()

            break


        if key == ord('p'):
            while 1:
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break


        if key == ord(' '):

            draw_board = np.ones((int(cover*3000),int(cover*3000),3),np.dtype('uint8'))*255








def show_record():
    '''展示历史记录3d'''

    num = input('record num:')
    record = np.load(f'3d_pos{num}.npz')
    pos_3d_x = record['pos_3d_x']
    pos_3d_y = record['pos_3d_y']
    pos_3d_z = record['pos_3d_z']

    
    pos_3d_x = np.array(pos_3d_x[pos_3d_x != np.array(None)])
    pos_3d_y = np.array(pos_3d_y[pos_3d_y != np.array(None)])
    pos_3d_z = np.array(pos_3d_z[pos_3d_z != np.array(None)])


    pos_3d_x = np.convolve(pos_3d_x,filter_window,'valid')
    pos_3d_y = np.convolve(pos_3d_y,filter_window,'valid')
    pos_3d_z = np.convolve(pos_3d_z,filter_window,'valid')


    nums = np.shape(pos_3d_x)[0]
    pos_2d_x=[]
    pos_2d_y=[]
    for i in range(nums):
        if pos_3d_z[i]<mindis:
            pos_2d_x.append(pos_3d_x[i])
            pos_2d_y.append(pos_3d_y[i])
        else:
            pos_2d_x.append(None)
            pos_2d_y.append(None)

    pos_2d_x = np.array(pos_2d_x)
    pos_2d_y = np.array(pos_2d_y)


   
    #二维绘图
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(1, 1, 1)

    ax2d.set_xlim(-cover/2,cover/2)
    ax2d.set_ylim(-cover/2,cover/2)
    ax2d.plot(pos_2d_x,pos_2d_y,label='captured 2d curve')
    ax2d.legend()


    #三维绘图
    fig3d = plt.figure()
    ax3d = fig3d.gca(projection='3d')

    ax3d.set_xlim(-cover/2,cover/2)
    ax3d.set_ylim(-cover/2,cover/2)
    ax3d.set_zlim(-cover/2,cover/2)
    ax3d.plot(pos_3d_x,pos_3d_y,pos_3d_z,label='captured 3d curve')
    ax3d.legend()

    plt.show()
    


def get_min_z():
    pre_rs,pre_ts=offline_run()
    poses=[]
    min_s=1000000
    min_z=np.array([[0,0,-pl]]).T
    for i in range(-50,51,1):
        for j in range(-50,51,1):
            for l in range(1630,1641,1):
                z = np.array([[i/10000.0,j/10000.0,-l/10000.0]]).T
                poses=[]
                for (pre_r,pre_t) in zip(pre_rs,pre_ts):
                    R,jacobin = cv2.Rodrigues(pre_r)
                    pos = np.dot(R,z).T+pre_t#计算笔尖坐标
                    pos = np.dot(tran_mtx, (pos-trans_tvec).reshape(3,1)).T
                    poses.append(pos)
                poses=np.array(poses)
                

                posx=poses[:,0,0]
                posy=poses[:,0,1]
                sx = np.max(posx)-np.min(posx)
                sy = np.max(posy)-np.min(posy)
                s = (sx+sy)/2.0
                if s<min_s:
                    min_s=s
                    min_z=z.copy()
    
    print("min_z:\n",min_z)
    print("min_s: ",min_s)

        

#online_run()
#get_video()
#offline_run()
#show_record()
#get_min_z()


if bot_enable:
    api = dType.load()
    #建立与dobot的连接
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        online_run()
        #断开连接
        #Disconnect Dobot
        dType.DisconnectDobot(api)
else :
    online_run()