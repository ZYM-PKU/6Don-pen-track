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


#基本参数设定
a = 0.025 #正方体棱长2.5cm
pl = 0.134 #笔长13.4cm
d = 0.022 #marker边长2.2cm
mindis = 0.001 #最小笔尖距1mm


rgb = 0#是否采用彩色模式
top_id = 1#顶端标签
up_ids=[2,3,4,5]#上层标签
cap_para = {'choice':0,'w':2592,'h':2048,'fps':40}


cover = 0.3#画布范围（单位：m）
draw_board = np.ones((int(cover*3000),int(cover*3000),3),np.dtype('uint8'))*255#画布

window_size = 8
filter_window= np.ones(window_size,dtype=np.float)/window_size #滑动窗口
font = cv2.FONT_HERSHEY_SIMPLEX #字体


#  mtx:
#  [[2.61585666e+03 0.00000000e+00 1.23637416e+03]
#  [0.00000000e+00 2.61553224e+03 1.04108063e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

#标定与校准
mtx,dist,newcameramtx,roi = get_camera_mtx(0) #相机内参
print("mtx:\n",mtx)
print("dist:\n",dist)

r_mtx , t_mtx = get_marker_mtx(cap_para,0) #立方体参数

tip_tvec = get_tip_tvec(cap_para,0).reshape((3,1)) #笔尖向量

trans_mtx, trans_tvec= get_trans_mtx(cap_para,0) #世界坐标系转换


 

def get_video():
    '''视频录制'''

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



def run(online = 1, draw = 1):
    '''实时/离线轨迹检测'''

    global draw_board

    if online:
        cam, stFrameInfo, nPayloadSize = init_cam()
    else:
        i = int(input("video id:"))
        vcap = cv2.VideoCapture(f'output{i}.avi') 

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

    if draw:
        cv2.namedWindow('board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('board', int(cover*3000),int(cover*3000))
    
    
    #时间
    tic = time.time()-1

    #3d坐标列
    pos_3d_x=np.array([])
    pos_3d_y=np.array([])
    pos_3d_z=np.array([])


    #缓存信息
    last_2d_x = None
    last_2d_y = None#实时绘制数据保存

    frame_slice = np.array([0,2048,0,2592])
    old_gray=np.array([])
    prev_corners=np.array([])
    old_ids = np.array([])


    while True:

        #帧率测试
        toc = time.time()
        time_dure = toc-tic
        current_fps = 1 / time_dure 
        tic = time.time()

        cv2.putText(frame, f"fps:  {current_fps}", (2592-400,64), font, 1, (0,0,255),2,cv2.LINE_AA)


        #读取帧内容
        if online:
            ret, frame = cam_read(cam, stFrameInfo, nPayloadSize, rgb)
        else:
            try:
                ret, frame = vcap.read()
                h, w = frame.shape[:2]

            except:

                num = int(input("record num:"))
                np.savez(f'3d_pos{num}.npz',pos_3d_x=pos_3d_x,pos_3d_y=pos_3d_y,pos_3d_z=pos_3d_z)
                vcap.release()
                cv2.destroyAllWindows()

                break



        #ROI目标追踪
        corners,ids,frame_slice,old_gray,prev_corners,old_ids = Target_tracking(frame_slice,frame,rgb,old_gray,prev_corners,old_ids)


        #执行主算法
        if ids is not None: 

            corners = np.array(corners) #corners: array(n,1,4,2)
            ids = ids.ravel('F').tolist()
            

            #位姿检测
            rvec, tvec = estimatePoseMarkers_Z(corners, d, mtx, dist, ids, r_mtx, t_mtx) #计算顶端坐标系
            #rvec1 , tvec1 = GN(corners,rvec1,tvec1,d,3,1)#高斯牛顿迭代
            #rvec1 , tvec1 = calibrate_poses(rvec1,tvec1,corners,mtx,d,r_lr,t_lr,min_loss,max_iters)#梯度下降


            #绘制坐标轴
            R,jacobin = cv2.Rodrigues(rvec)

            for i in range(ids):
                
                R_mtx = cv2.Rodrigues(r_mtx[i])
                temp_R = np.dot(R_mtx, R)
                temp_rvec = cv2.Rodrigues(temp_R)
                temp_tvec = tvec + np.dot(R,t_mtx[i].T).T

                aruco.drawAxis(frame, mtx, dist, temp_rvec, temp_tvec, d)
                aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            cv2.rectangle(frame, (frame_slice[2],frame_slice[0]),(frame_slice[3],frame_slice[1]),(0,0,255),2)


            #计算笔尖
            rvec = rvec_reverse(rvec)  #反转Z轴，指向笔尖
            pos = calc_pos(rvec, tvec, tip_tvec)#计算笔尖坐标
            pos = trans_pos(pos, trans_mtx, trans_tvec) #转换到世界坐标系
            print(pos)
            

            #绘制轨迹曲线
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
                else:
                    new_2d_x = None
                    new_2d_y = None

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
        

        else:
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    


        # 显示结果  
        temp = cv2.resize(frame,(2592//2,2048//2))
        cv2.imshow("frame",temp)
        if draw: cv2.imshow("board",draw_board)
        key = cv2.waitKey(1)


        # 按esc键退出
        if key == 27:   

            num = int(input("record num:"))
            np.savez(f'3d_pos{num}.npz',pos_3d_x = pos_3d_x, pos_3d_y = pos_3d_y, pos_3d_z = pos_3d_z)
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
    


#get_video()
run()
