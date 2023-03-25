import numpy as np
import cv2
import glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
def load_image_points(left_dir, left_prefix, right_dir, right_prefix, image_format, square_size, width=9, height=6):
    global image_size
    pattern_size = (width, height)  # Chessboard size!
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    # Left directory path correction. Remove the last character if it is '/'
    if left_dir[-1:] == '/':
        left_dir = left_dir[:-1]

    # Right directory path correction. Remove the last character if it is '/'
    if right_dir[-1:] == '/':
        right_dir = right_dir[:-1]

    # Get images for left and right directory. Since we use prefix and formats, both image set can be in the same dir.
    left_images = glob.glob(left_dir + '/' + left_prefix + '*.' + image_format)
    right_images = glob.glob(right_dir + '/' + right_prefix + '*.' + image_format)

    # Images should be perfect pairs. Otherwise all the calibration will be false.
    # Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
    # Sort will fix the globs to make sure.
    left_images.sort()
    right_images.sort()

    pair_images = zip(left_images, right_images)  # Pair the images for single loop handling

    # Iterate through the pairs and find chessboard corners. Add them to arrays
    # If openCV can't find the corners in one image, we discard the pair.
    num = 0
    for left_im, right_im in pair_images:
        if num>1:
            continue
        num+=1
        # Right Object Points
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
        # Left Object Points
        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
        # corners_left.sort()
        # corners_right.sort()
        for i in range(len(corners_left)):
            
            left = cv2.circle(left,(int(corners_left[i][0][0]),int(corners_left[i][0][1])),radius=2, color=(0, 0, 255), thickness=-1)
            right = cv2.circle(right,(int(corners_right[i][0][0]),int(corners_right[i][0][1])),radius=2, color=(0, 255, 0), thickness=-1)
        # gray_left=cv2.drawChessboardCorners(gray_left,(9,6),corners_left,ret_left)
            cv2.imshow("t",np.hstack([left,right]))
            cv2.waitKey(0)

        if ret_left and ret_right:  # If both image is okay. Otherwise we explain which pair has a problem and continue
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
            l_c = np.asanyarray(corners_left).reshape([54,2])
            r_c = np.asanyarray(corners_right).reshape([54,2])
            # l_c = np.sort(l_c,0)
            # l_c = np.sort(l_c,1)
            # r_c = np.sort(r_c,0)
            # r_c = np.sort(r_c,1)
            l_c = np.round(l_c)
            r_c = np.round(r_c)
            l_c = l_c.astype(np.int32)
            r_c = r_c.astype(np.int32)
            res_s = ""
            for i in range(54):
                res_s = res_s + f"{l_c[i][0]} {l_c[i][1]} {r_c[i][0]} {r_c[i][1]}\n"
            
            print(res_s)
            open("data/6/match.txt",'w').write(res_s)
            # print(corners2_left[:][0],corners2_right[:][0])
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue

    # print(left_imgpoints[0][0][0])
    image_size = gray_right.shape  # If you have no acceptable pair, you may have an error here.
    return [objpoints, left_imgpoints, right_imgpoints]

right_img_dir  = "data/6"
left_img_dir = "data/6"
load_image_points( left_img_dir, "imgL", right_img_dir,"imgR", "png", 0.35)