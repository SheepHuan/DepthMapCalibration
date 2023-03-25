import cv2
import numpy as np
import copy
from sift import SIFT_Match
# from corner_detect import shi_tomas
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
        copyed = cv2.putText(copyed, fr"{depth_map[y][x]}", (0, showing_img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255),2)
        # point_size = 1
        # point_color = (0, 0, 255) # BGR
        # thickness = 4 # 可以为 0 、4、8
        # copyed = cv2.circle(copyed, (x,y), point_size, point_color, thickness)
        cv2.imshow(windowsName,copyed)
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(windowsName,x,y,end='\n')
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        showing_img = cv2.circle(showing_img, (x,y), point_size, point_color, thickness)

MAX_DEPTH = 2000
MIN_DEPTH = 0
def getDepthFromTxt(filename,rows=180,cols=240,channles=1):
    depth = np.zeros((180,240),dtype = np.uint16)
    confidence = np.zeros((180,240))
    lines = open(filename, 'rt').read().split("\\n")
    lines = [line for line in lines if line!=""]
    j = 0
    for line in lines:
        l = line.split('\\t')[:-1]
        l = [int(i) for i in l]
        data_line = l[:240]
        conf_line = l[240:480]
        depth[j,:] = data_line
        confidence[j,:] = conf_line
        j = j + 1
    depth[depth == np.Inf] = MIN_DEPTH
    depth[depth > MAX_DEPTH] = MIN_DEPTH
    return depth, confidence

def getDepthFromRaw(filename,rows=480,cols=640,channels=1):
    img=np.fromfile(filename, dtype='uint16')
    # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
    depth =img.reshape(rows, cols, channels)
    depth[depth == np.Inf] = MIN_DEPTH
    depth[depth > MAX_DEPTH] = MIN_DEPTH
    return depth


def filterDepth(src,min,max,val):
    src[np.logical_and(src > min,src < max)] = val
    src[~np.logical_and(src > min,src < max)] = 0
    return src

def filterSmallAreas(img,s=1,e=5):
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # print('num_labels = ',num_labels)
    # # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    # print('stats = ',stats)
    # # 连通域的中心点
    # print('centroids = ',centroids)
    # # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    # print('labels = ',labels)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(s, e):
        mask = labels == i
        r,g,b = 255,255,255
        output[:, :, 0][mask] = b
        output[:, :, 1][mask] = g
        output[:, :, 2][mask] = r
    return output

def conersDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 数据类型转换成float32
    gray_float32 = np.float32(gray)

    # 角点检测
    dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)

    #设置阈值,将角点绘制出来,阈值根据图像进行选择
    R=dst.max() * 0.01
    #这里将阈值设为dst.max()*0.01 只有大于这个值的数才认为数角点
    img[dst > R] = [0, 0, 255]
    # cv2.imshow("conners",img)
    return img

color_list=[
    (0,255,0),
    (255,0,0),
    (0,0,255),
    (255,255,0),
    (255,0,255)
]
def det_five_pointed_star(img):
    """
    检测五角星，得到其周长和面积
    args:
       img: img_numpy
    returns:
        if have five_pointed_star:
          perimeter and area
        else:
            None
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("out",binary_img)
    # cv2.waitKey(0)
    all_pts=[]
    for i,contour in enumerate(contours):
        # idx = contours_l.index(max(contours_l))
        # 计算五角星轮廓的周长,然后将其缩小，作为下一步的阈值
        perimeter = cv2.arcLength(contour, True)
        ep = 0.02 * perimeter 
        # 轮廓的转折点的近似点
        approx = cv2.approxPolyDP(contour, ep, True)
        # print(approx)
        img = cv2.drawContours(img, approx, -1, color_list[i%5], 2)
        area_star = cv2.contourArea(approx)
        co = len(approx)
        print('*** num of point ***', co)
        if co == 10:
            pts = []
            for pt in approx:
                pts.append([pt[0][0],pt[0][1]])
            # print(pts)
        # 先五角星排序
            all_pts.append(sorted(pts,key= lambda x: (x[0],x[1])))
    # print(1)
    all_pts = sorted(all_pts,key= lambda x: (x[0][0],x[0][1]))

    return img,all_pts

# cv2.namedWindow("out",cv2.WINDOW_NORMAL)
# cv2.namedWindow("out")
tof_raw_depth,tof_raw_confidence = getDepthFromTxt(r"D:\code\MobileToFAndStereo\app\src\main\proto\1679666947511.txt")
realsense_raw_depth = getDepthFromRaw("data/5/rs_Depth.raw")
realsense_raw_depth = np.rot90(realsense_raw_depth,2)
realsense_raw_depth = cv2.resize(realsense_raw_depth,(tof_raw_depth.shape[1],tof_raw_depth.shape[0]),None)

realsense_raw_left = cv2.imread("rs_bin.png")
# realsense_raw_left = np.rot90(realsense_raw_left,2)
realsense_raw_left = cv2.resize(realsense_raw_left,(tof_raw_depth.shape[1],tof_raw_depth.shape[0]),None)

# out = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,pos,1)

tof_raw_depth = filterDepth(tof_raw_depth,390,520,480)
# realsense_raw_depth = filterDepth(realsense_raw_depth,400,550,480)

img_color_tof = cv2.applyColorMap(visDisp(tof_raw_depth),cv2.COLORMAP_PARULA) 
img_color_real = cv2.applyColorMap(visDisp(realsense_raw_depth),cv2.COLORMAP_PARULA) 

tof_out = filterSmallAreas(copy.deepcopy(img_color_tof))
real_out = filterSmallAreas(copy.deepcopy(realsense_raw_left),3,7)

res_tof,pts_tof = det_five_pointed_star(copy.deepcopy(tof_out))
res_rs,pts_rs = det_five_pointed_star(copy.deepcopy(real_out))
for i in range(4):
    for j in range(10):
        print(*pts_tof[i][j],*pts_rs[i][j])
cv2.imshow("out",np.hstack([res_tof,res_rs]))

# cv2.imshow("img_tof",img_color_tof)
# cv2.imwrite("img_rs",real_out)

# cv2.imshow("l:tof, r:gt",np.hstack([img_color_tof,img_color_real]))
cv2.waitKey(0)






