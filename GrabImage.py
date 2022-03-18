# -- coding: utf-8 --

import sys
import threading
import msvcrt
import cv2,os
import numpy as np


from ctypes import *

PATH = os.path.dirname(__file__)
os.chdir(PATH)
sys.path.append("MvImport")

from MvImport.MvCameraControl_class import *
from CamOperation_class import *





def cam_read(cam,stFrameInfo,nPayloadSize,rgb=False):
    '''读取摄像机视频流，产生opencv图像'''
    
    if rgb:
        data_buf = (c_ubyte * (nPayloadSize *3))()
        ret = cam.MV_CC_GetImageForBGR(byref(data_buf), nPayloadSize*3, stFrameInfo, 1000)
    else:
        data_buf = (c_ubyte * nPayloadSize)()
        ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stFrameInfo, 1000)

    if ret == 0:
        temp = np.asarray(data_buf)  # 将c_ubyte_Array转化成ndarray
        if rgb:
            temp = temp.reshape((2048, 2592, 3))# 根据自己分辨率进行转化
        else:
            temp = temp.reshape((2048, 2592))# 根据自己分辨率进行转化
        #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)  # 这一步获取到的颜色不对，因为默认是BRG，要转化成RGB，颜色才正常
        return ret,temp

    else:
        raise RuntimeError("databuf interrepted")






def init_cam():

    cam = MvCamera()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    nPayloadSize=0


    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = 0

    assert nConnectionNum < deviceList.nDeviceNum
    
    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()



    #设置相机参数
    # ret1 = cam.MV_CC_SetFloatValue("ExposureTime",float(30000))
    # ret2 = cam.MV_CC_SetFloatValue("Gain",float(0))
    # ret3 = cam.MV_CC_SetFloatValue("AcquisitionFrameRate",float(70)) 
    # ret4 = cam.MV_CC_SetFloatValue("AcquisitionBitRate",float(128)) 
    # ret5 = cam.MV_CC_SetIntValue("Width",1000)
    # ret6 = cam.MV_CC_SetIntValueEx("Width",100)
    # ret7 = cam.MV_CC_SetEnumValue("nWidth",100)

    #ch:将相机属性导出到文件中 | en:Export the camera properties to the file
    # ret = cam.MV_CC_FeatureSave("FeatureFile.ini")
    # if MV_OK != ret:
    #     print ("save feature fail! ret [0x%x]" % ret)
    # print ("finish export the camera properties to the file")

    # print ("start import the camera properties from the file")
    # print ("wait......")

    #ch:从文件中导入相机属性 | en:Import the camera properties from the file
    ret = cam.MV_CC_FeatureLoad("FeatureFile.ini")
    if MV_OK != ret:
        print ("load feature fail! ret [0x%x]" % ret)
    print ("finish import the camera properties from the file")

    stBool = c_bool(False)
    ret =cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
    

    # ch:设置U3V的传输通道个数
    ret = cam.MV_USB_SetTransferWays(1)

    if ret != 0:
        print ("set transfer ways fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print ("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
        
    nPayloadSize = stParam.nCurValue


    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()


    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    return cam,stFrameInfo,nPayloadSize







def stop_cam(cam):

    '''停止采集'''
    ret = cam.MV_CC_StopGrabbing()
    ret = cam.MV_CC_CloseDevice()
    ret = cam.MV_CC_DestroyHandle()




if __name__ == "__main__":
    cam,stFrameInfo,nPayloadSize = init_cam()
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    #cv2.resizeWindow('frame', 2592//2, 2048//2)
    while True:
        ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,0)
        temp = cv2.resize(frame,(2592//2,2048//2))
        cv2.imshow('frame',temp)
        key = cv2.waitKey(1)

        if key == 27:
            stop_cam(cam)
            break




    
