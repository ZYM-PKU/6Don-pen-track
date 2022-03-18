#test my draw
import DobotDllType as dType
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}




window_size = 8
filter_window= np.ones(window_size,dtype=np.float)/window_size #滑动窗口

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


mindis=1
nums = np.shape(pos_3d_x)[0]
pos_2d_x=[]
pos_2d_y=[]
for i in range(nums):
    if pos_3d_z[i]<(mindis/1000):
        pos_2d_x.append(pos_3d_x[i])
        pos_2d_y.append(pos_3d_y[i])
    else:
        pos_2d_x.append(None)
        pos_2d_y.append(None)

pos_2d_x = np.array(pos_2d_x)
pos_2d_y = np.array(pos_2d_y)
#二维绘图
cover = 0.3
fig2d = plt.figure()
ax2d = fig2d.add_subplot(1, 1, 1)

ax2d.set_xlim(-cover/2,cover/2)
ax2d.set_ylim(-cover/2,cover/2)
ax2d.plot(pos_2d_x,pos_2d_y,label='captured 2d curve')
ax2d.legend()
plt.show()

def draw():
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
    # dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    
    print("enable rail is ",dType.GetDeviceWithL(api))

    pos = dType.GetPose(api)
    print(pos)
    X=pos[0]
    Y=pos[1]
    Z=pos[2]
    rHead = pos[3]
    
    L = 300
    all_y_move = 0

    nums = np.shape(pos_3d_x)[0]
    flag=True

    for i in range(nums):
        x=pos_3d_x[i]*1000*0.3
        y=pos_3d_y[i]*1000*0.3
        z=pos_3d_z[i]*1000
        if z>mindis:
            flag=True
            continue 
        if flag:

            lastIndex = dType.SetPTPWithLCmd(api,0,X-y,Y+x,Z,rHead,L,isQueued=1)[0]
            flag=False
        else:
            lastIndex = dType.SetPTPWithLCmd(api,2,X-y,Y+x,Z,rHead,L,isQueued=1)[0]

        if(i):all_y_move += pos_3d_y[i]-pos_3d_y[i-1]
        L = L - all_y_move
    

    print("exec")
    dType.SetQueuedCmdStartExec(api)
        #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)

    #停止执行指令
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)




api = dType.load()
#建立与dobot的连接
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])

if (state == dType.DobotConnect.DobotConnect_NoError):
    draw()
    #断开连接
    #Disconnect Dobot
    dType.DisconnectDobot(api)