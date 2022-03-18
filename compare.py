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


def histogram_demo(image1,image2):
    ax=plt.subplot(2,1,1)
    ax.hist(image1.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
    bx=plt.subplot(2,1,2)
    bx.hist(image2.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
    plt.show()




#指定工作目录
PATH = os.path.dirname(__file__)
os.chdir(PATH)

cam,stFrameInfo,nPayloadSize = init_cam()
frames=[]

for i in range(2):

    ret ,frame = cam_read(cam,stFrameInfo,nPayloadSize,i)
    if i==1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)

temp0 = frames[0].astype(float)
temp1 = frames[1].astype(float)



max0=np.max(temp0)
max1=np.max(temp1)
#histogram_demo(temp0,temp1)

d=temp0-temp1
maxd=np.max(d)
mind=np.min(d)
mean = np.mean(d)

cv2.imwrite("gray.jpg",temp0)
cv2.imwrite("rgbt.jpg",temp1)

