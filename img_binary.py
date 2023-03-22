
import cv2
import numpy as np
import copy
from sift import SIFT_Match


realsense_raw_left = cv2.imread("data/5/rs_left.png")
realsense_raw_left = np.rot90(realsense_raw_left,2)
img_gray = cv2.cvtColor(realsense_raw_left, cv2.COLOR_BGR2GRAY )
windowsName = "BIN"

# def onChangeTrackBar(pos):
  
#     # ret,out = cv2.threshold(img_gray,pos,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#     # out = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,pos,1) #自适应高斯加权
#     if pos % 2 == 0:
#         pos+=1
#     out = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,pos,1)
#     cv2.imshow(windowsName,out)

# cv2.namedWindow(windowsName,cv2.WINDOW_AUTOSIZE)
# cv2.createTrackbar("threshold",windowsName,0,800,onChangeTrackBar)

# cv2.waitKey(0)

out = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,801,1)
cv2.imwrite("rs_bin.png",out)





