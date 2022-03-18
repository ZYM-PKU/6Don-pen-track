#test my draw
import DobotDllType as dType
import numpy as np
import math
import threading
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

SCALE_factor = 0.2
autoz = 23
L_movest = 400
x_st = 208
y_st = 0#相对于L的
npz_line_index = 1
npz = np.load("shilong_data_210325_modified_processed_rdp2.npz",allow_pickle=True)


all_words = npz["all_words"]
tag_gbk_str_lists = npz["tag_gbk_str_lists"]
char_point_num_accum_lists = npz["char_point_num_accum_lists"]

print(len(all_words))#7kpoints
print(all_words[0:18])
print(len(tag_gbk_str_lists))#第一个npz file是8行
print(tag_gbk_str_lists)#[list(['gbk words in a sen'])]
print(len(char_point_num_accum_lists ))#8
print(char_point_num_accum_lists)#[list([character end point index in a sen])]
#每个字写完之后调传送带
end_ls = char_point_num_accum_lists[npz_line_index]#测试第一个list
last = end_ls[-1]#last index之前的为属于本行的点

# import sys
# sys.exit(0)
#将dll读取到内存中并获取对应的CDLL实例
#Load Dll and get the CDLL object
api = dType.load()
#建立与dobot的连接
#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])

if (state == dType.DobotConnect.DobotConnect_NoError):

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
    posL = dType.GetPoseL(api)
    #调零完成
    print(pos,posL)#[143.50733947753906, -2.5858712196350098, 134.75035095214844, 可以小数
    # x = pos[0]#100-360.归到100？
    # y = pos[1]#0-1000
    # z = pos[2]#-62是目前的autoz
    # x= 110#repos x
    rHead = pos[3]
    # L = posL[0]
    x = x_st
    #y在studio里面是相对于L的
    y = y_st
    z = autoz
    pre_last = char_point_num_accum_lists[npz_line_index-1][-1] if npz_line_index!=0 else 0
    st_pre = pre_last
    #挪动一下
    #test L
    L = L_movest
    z = autoz
    #jump 2 start point
    lastIndex = dType.SetPTPWithLCmd(api,0,x,y,z,rHead,L,isQueued=1)[0]
    posL = dType.GetPoseL(api)
    #队列长度有限？
    for i in range(len(end_ls[:6])):
        #绘制单个字
        last_dot = end_ls[i]
        print(last_dot)
        all_y_move = 0#根据字移动机械臂
        #pre last
        for j in range(pre_last,last_dot):
            #draw a line
            if j == st_pre:#起点设置好了
                continue
            pre_dot = all_words[j-1]
            tmp_dot = all_words[j]
            all_y_move += tmp_dot[0]*SCALE_factor

            x = x+ tmp_dot[1]*SCALE_factor
            y = y+ tmp_dot[0]*SCALE_factor
            if pre_dot[-1]==0:#上一个不是终结点，继续画
                #drwa tmp dot
                #和dot的xy相反
                print("line for ",tmp_dot,pre_dot,x,y)
                lastIndex = dType.SetPTPWithLCmd(api,2,x,y,z,rHead,L,isQueued=1)[0]#add2que
            else:#jump,到了下一笔
                lastIndex = dType.SetPTPWithLCmd(api,0,x,y,z,rHead,L,isQueued=1)[0]#add2que
        #process arm move
        L = L - all_y_move#移动
        y -= all_y_move
        #这个是看一下是不是点的开局的y的位置
        # lastIndex = dType.SetPTPWithLCmd(api,0,x,y,z,rHead,L,isQueued=1)[0]

        pre_last = last_dot

    print("exec")
    dType.SetQueuedCmdStartExec(api)
        #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)

    #停止执行指令
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)

#断开连接
#Disconnect Dobot
dType.DisconnectDobot(api)