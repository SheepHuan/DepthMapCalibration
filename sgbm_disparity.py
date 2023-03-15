import cv2
import numpy as np
import copy


def wls_l(img,disp1,disp2,mather):
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=mather)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filteredImg = wls_filter.filter(disp1, img, None, disp2)  # important to put "imgL" here!!!
    filteredImg = np.clip(filteredImg, 0, None)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def visDisp(disp):
    filteredImg = np.clip(disp, 0, None)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def compute_disparity(imgL,imgR):
    window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities= 12 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=7,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH4
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
   
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
   
    # filteredImg1 = wls_l(imgL,displ,dispr,left_matcher)
    # filteredImg2 = wls_r(imgR,dispr,displ,right_matcher)
    filteredImg1 = visDisp(displ)
    filteredImg2 = visDisp(dispr)
    # unnormalized_filteredImg = copy.deepcopy(filteredImg)
    #
    #
    #

    # return displ, dispr
    return filteredImg1, filteredImg2


if __name__=="__main__":
    imgL =  cv2.imread("data/Flowerpots/view0.png",-1)
    imgR1 = cv2.imread("data/Flowerpots/view1.png", -1)
    # imgR2 = cv2.imread("data/Flowerpots/view2.png", -1)
    # imgR3 = cv2.imread("data/Flowerpots/view3.png", -1)

    displ,dispr = compute_disparity(imgL,imgR1)
    cv2.imshow("t",np.hstack((displ,dispr)))
    cv2.waitKey(0)
    # res2,_ = compute_disparity(imgL,imgR2)
    # res3,_ = compute_disparity(imgL,imgR3)

    # print(res1[1000:1010,1000:1010])
    # print(res2[1000:1010,1000:1010])
    # print(res3[1000:1010,1000:1010])
    # cv2.imshow("1",res1)
    # cv2.imshow("2",res2)
    # cv2.waitKey()