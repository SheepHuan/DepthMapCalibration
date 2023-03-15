import cv2
import numpy as np
from sgbm_disparity import compute_disparity

def SIFT_Match(img1,img2):
    sift = cv2.SIFT_create()
    # 获取关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1,None) 
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # 可视化点和匹配
    img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
    img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))
    vis1 = np.hstack([img3,img4])
    cv2.imshow("points",vis1)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("BFmatch", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    imgL =  cv2.imread("data/Flowerpots/view0.png",-1)
    imgR1 = cv2.imread("data/Flowerpots/view1.png", -1)
    imgR2 = cv2.imread("data/Flowerpots/view2.png", -1)
    img1,vis1 = compute_disparity(imgL,imgR1)
    img2,vis2 = compute_disparity(imgL,imgR2)
    SIFT_Match(img1,img2)