import numpy as np
import cv2
def shi_tomas(img):
    #shi-Tomas角点检测
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,50,0.5,10)
    print(corners)
    #绘制角点
    for corner in corners:
        x,y=corner.ravel()
        cv2.circle(img,(int(x),int(y)),2,(0,0,255),-1)
    cv2.imshow("shi-Tomas",img)