#本程序用于生成aruco标记
import cv2#4.4.0
import os,sys
import numpy as np

#指定工作目录
PATH = os.path.dirname(__file__)
os.chdir(PATH)

# 生成aruco标记
# 加载预定义的字典
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

markerImage = np.zeros((int(200*(2.2/5.29)), int(200*(2.2/5.29))), dtype=np.uint8)

for i in range(12):
    markerImage = cv2.aruco.drawMarker(dictionary, i+1, int(200*(2.2/5.29)), markerImage, 1)

    cv2.imwrite(f'markers/{i+1}.png', markerImage)
