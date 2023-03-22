import cv2
import numpy as np
import copy
from sift import SIFT_Match
def visDisp(disp):
    # filteredImg = np.clip(disp, 0, None)
    filteredImg = cv2.normalize(src=disp, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def drawPoints(imgO,pts):
    img = copy.deepcopy(imgO)
    for pt in pts:
        point_size = 1
        point_color = (255, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        img = cv2.circle(img, pt, point_size, point_color, thickness)
    return img

def drawLines(img1,img2,pts1,pts2):
    img = np.hstack([img1,img2])
    point_color = (0, 0, 255) # BGR
    thickness = 1
    lineType = 4
    for i,pt1 in enumerate(pts1):
        pt2 = pts2[i]
        pt2[0]+=img1.shape[1]
        print(pt1,pt2)
        img = cv2.line(img,pt1,pt2,point_color,thickness,lineType)
    return img
def on_mouse_moving(event, x, y, flags, param):
    depth_map, showing_img ,windowsName= param
    if event == cv2.EVENT_MOUSEMOVE:
        xy = "%d,%d" % (x, y)
        # print(x,y)
        copyed = copy.deepcopy(showing_img)
        # cv2.putText(copyed, fr"{depth_map[y][x]}", (0, showing_img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255),2)
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        copyed = cv2.circle(copyed, (x,y), point_size, point_color, thickness)
        cv2.imshow(windowsName,copyed)
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(windowsName,x,y,end='\n')
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        showing_img = cv2.circle(showing_img, (x,y), point_size, point_color, thickness)


img_tof = cv2.imread("img_tof.png",-1)
img_rs = cv2.imread("img_rs.png",-1)
vis1 = np.hstack([copy.deepcopy(img_tof),copy.deepcopy(img_rs)])
# cv2.imshow("before",)
# 单应性矩阵计算+透视变换
lines = open("match.txt").readlines()
lines = [line.replace("\n","") for line in lines]
tof_pts = [[int(line.split(' ')[0]),int(line.split(' ')[1])] for line in lines]
real_pts = [[int(line.split(' ')[2]),int(line.split(' ')[3])] for line in lines]

tof_pts_np = np.asarray(tof_pts)
real_pts_np = np.asarray(real_pts)
matrix, mask = cv2.findHomography(real_pts_np, tof_pts_np, 0)
print(f'matrix: {matrix}')
perspective_real = cv2.warpPerspective(img_rs, matrix, (img_tof.shape[1], img_tof.shape[0]))
vis2 = drawLines(drawPoints(img_tof,tof_pts),drawPoints(perspective_real,tof_pts),tof_pts,copy.deepcopy(tof_pts))
cv2.imshow("top: before, bottom: after",np.vstack([vis1,vis2]))
cv2.waitKey(0)





