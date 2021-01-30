import numpy as np
import cv2
import glob
import os
import pickle

def detect_corners(images, h_corners, v_corners):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((v_corners*h_corners,3), np.float32)
    objp[:,:2] = np.mgrid[0:h_corners,0:v_corners].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img_names = []

    for fname in images:
        img = cv2.imread(fname, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)

        img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h_corners,v_corners),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            img_names.append(fname)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (h_corners,v_corners), corners2,ret)

            cv2.imshow('img',img)
            cv2.waitKey(500)
        else:
            print(f'{fname} failed')

    cv2.destroyAllWindows()

    return gray.shape[::-1], objpoints, imgpoints, img_names


def calibrate_chessboard(size, objpoints, imgpoints):
    cameraMatrixInit = np.array([[ 1000.,    0., size[0]/2.],
                                 [    0., 1000., size[1]/2.],
                                 [    0.,    0.,           1.]])
    rpe, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, cameraMatrixInit, None, flags = cv2.CALIB_FIX_ASPECT_RATIO)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)
    return rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi


if os.path.exists('chessboard_corners.pkl'):
    with open('chessboard_corners.pkl', 'rb') as f:
        size, objpoints, imgpoints, img_names = pickle.load(f)
else:
    size, objpoints, imgpoints, img_names = detect_corners(glob.glob(r"C:\Users\stelli\Desktop\Photogrammetrie\data\basement_1\*.JPG"), 7, 4)
    with open('chessboard_corners.pkl', 'wb') as f:
        pickle.dump((size, objpoints, imgpoints, img_names), f)

if os.path.exists('chessboard_calib.pkl'):
    with open('chessboard_calib.pkl', 'rb') as f:
        rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi = pickle.load(f)
else:
    rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi = calibrate_chessboard(size, objpoints, imgpoints)
    with open('chessboard_calib.pkl', 'wb') as f:
        pickle.dump((rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi), f)

print( f"Reprojectionerror reported by cv2.calibrateCamera(): {rpe}")

print(size)
print(mtx)
print(dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
    print(error)
print( f"total error: {mean_error/len(objpoints)}")



# mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, shape, cv2.CV_32FC1)
#
# # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# for img_name in images:
#     img = cv2.imread(img_name, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
#     dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#     cv2.imshow('img', dst)
#     if cv2.waitKey(-1) & 0xFF == ord('q'):
#         break


# todo stl: this makes no sense at all! cv2.findChessboardCorners doesn't work on high resolution images !!! It end up in an endlessloop.
#  Scaling down by 4 helps and all but two images succeed!
#   STL04007.JPG extrem verzerrt in der Ecke aber komplett
#   STL04031.JPG mittig aber fast Bild-füllend
#  Scaling down by 8
#   STL04012.JPG
#   STL04016.JPG


