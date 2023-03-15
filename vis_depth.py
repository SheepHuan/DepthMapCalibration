import cv2
import numpy as np
import copy
from sift import SIFT_Match
from corner_detect import shi_tomas
def visDisp(disp):
    # filteredImg = np.clip(disp, 0, None)
    filteredImg = cv2.normalize(src=disp, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

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


tof_raw_depth = cv2.imread("data/re_tof_4/raw_tof_depth.png",cv2.IMREAD_ANYDEPTH)
tof_raw_depth = np.rot90(tof_raw_depth,1)
realsense_raw_depth = cv2.imread("data/re_tof_4/2023_03_15_17_01_35_realsense_depth.png",cv2.IMREAD_ANYDEPTH)
realsense_raw_depth = cv2.resize(realsense_raw_depth,(tof_raw_depth.shape[1],tof_raw_depth.shape[0]),None)


img_color_tof = cv2.applyColorMap(visDisp(tof_raw_depth),cv2.COLORMAP_PARULA) 
img_color_real = cv2.applyColorMap(visDisp(realsense_raw_depth),cv2.COLORMAP_PARULA) 

# 点标定
# cv2.imshow("tof",img_color_tof)
# cv2.setMouseCallback('tof', on_mouse_moving,[img_color_tof,img_color_tof,'tof2'])
# cv2.imshow("real",img_color_real)
# cv2.setMouseCallback('real', on_mouse_moving,[img_color_real,img_color_real,'real2'])


# 单应性矩阵计算+透视变换
# lines = open("data/re_tof_4/matched.txt").readlines()
# lines = [line.replace("\n","") for line in lines]
# tof_pts = [(int(line.split(' ')[0]),int(line.split(' ')[1])) for line in lines]
# real_pts = [(int(line.split(' ')[2]),int(line.split(' ')[3])) for line in lines]
# tof_pts = np.asarray(tof_pts)
# real_pts = np.asarray(real_pts)
# print(tof_pts)
# print(real_pts)

# matrix, mask = cv2.findHomography(real_pts, tof_pts, 0)
# print(f'matrix: {matrix}')
# perspective_img = cv2.warpPerspective(img_color_real, matrix, (img_color_tof.shape[1], img_color_tof.shape[0]))
# cv2.imshow("t",np.hstack([img_color_tof,perspective_img]))
# cv2.waitKey(0)





