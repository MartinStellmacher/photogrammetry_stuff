import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd


img_path = r"C:\Users\stelli\Desktop\Photogrammetrie\data\basement_1\*.JPG"

dict_marker = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
dict_board = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# todo: move to a separate file
def detect_and_dump(img, dict):
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dict)
    if ids is None:
        print('\tNone')
        return 0
    else:
        cv2.aruco.drawDetectedMarkers( img, corners, ids)
        print(f'\t{len(ids)}\n\t\t{sorted(ids.ravel().tolist())}')
        return len(ids)


# todo: move to a separate file
def check_all_images():
    images = glob.glob(img_path)
    for img in images:
        print(img)
        img = cv2.imread(img, flags=cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_GRAYSCALE)

        detect_and_dump(img, dict_marker)
        detect_and_dump(img, dict_board)

        out = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
        cv2.imshow('markers', out)
        cv2.waitKey(-1)

    cv2.destroyWindow('markers')


# todo: return DataFrame
# todo: assert that all images are of same size
def read_chessboards(images, board, dict):
    allCorners = []
    allIds = []

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im, flags=cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_GRAYSCALE)
        # first detect the markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dict)
        # then find the chessboard corners
        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(frame, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001))
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,frame,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
                allCorners.append(res2[1])
                allIds.append(res2[2])
                if len( board.chessboardCorners) != res2[0]:
                    print(f'\tdetected only {res2[0]} of {board.chessboardCorners} chessboard corners')
            else:
                print('\tfailed')

    imsize = frame.shape[::-1]
    return allCorners, allIds, imsize

# todo: second path without 10 worst images
# todo: return Dataframe
def calibrate_camera(allCorners, allIds, imsize, board):
    # fx == fy, absolute value has no meaning if CALIB_FIX_ASPECT_RATIO is specified
    # assume perfectly centered optics for the start
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    # flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    flags = (cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print(f'optics decentered {(np.divide(camera_matrix[0:2,2],np.array(imsize)/2)-1)*100} [%]')
    print(f'optics decentered {camera_matrix[0:2,2] - np.array(imsize)/2} [Pixel]')


    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
    # ret is ((perViewErrors*perViewErrors).sum()/len(perViewErrors))**0.5
    # [1], [3x3], [5], [nImgx3x1], [nImgx3x1]
    # stdDeviationsIntrinsics [2+2+5]
    # stdDeviationsExtrinsics [nImgx2x3]
    # perViewErrors [nImg]
    # pandas result: [nImg x (rvec, tvec, std_dev_ext, per_view_errors)],



def show_undistortion( img_name, mtx, dist):
    plt.figure()
    frame = cv2.imread(img_name)
    img_undist = cv2.undistort(frame, mtx, dist, None)
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()

# check_all_images()

# Markergröße Pixel: 731 [m]: 0.0619
# Kachelgröße Pixel: 1045 [m]: 0.0885
charuco_board = cv2.aruco.CharucoBoard_create(5, 8, 0.0885, 0.0619, dict_board)
images = glob.glob(img_path)

if os.path.exists( 'charuco_corners.pkl'):
    with open('charuco_corners.pkl', 'rb') as f:
        allCorners, allIds, imsize = pickle.load( f)
else:
    allCorners, allIds, imsize = read_chessboards(images, charuco_board, dict_board)
    with open('charuco_corners.pkl', 'wb') as f:
        pickle.dump( (allCorners, allIds, imsize), f)

if os.path.exists( 'charuco_calibration_xxx.pkl'):
    with open('charuco_calibration.pkl', 'rb') as f:
        ret, mtx, dist, rvecs, tvecs = pickle.load( f)
else:
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize, charuco_board)
    with open('charuco_calibration.pkl', 'wb') as f:
        pickle.dump( (ret, mtx, dist, rvecs, tvecs), f)

print(imsize)
print(np.array(imsize)/2)
print(ret)
print(mtx)
print(dist.ravel().tolist())


def reprojection_error( img_points, obj_points, rvecs, tvecs, imsize):
    heatmap = np.zeros(imsize, np.float32)
    err_vecs = None
    errors = []
    for img_point, rvec, tvec in zip(img_points, rvecs, tvecs):
        img_point2, _ = cv2.projectPoints( obj_points, rvec, tvec, mtx, dist)
        error = cv2.norm(img_point, img_point2, cv2.NORM_L2) / len(img_point2)
        errors.append( error)
        for ip in img_point:
            heatmap[tuple(ip.round().astype(int)[0])] += 1
        err = (img_point2 - img_point)
        ip_err = np.concatenate([img_point, err], axis=1)
        if err_vecs is None:
            err_vecs = ip_err
        else:
            err_vecs = np.concatenate([err_vecs, ip_err])
    max_err = np.linalg.norm(err_vecs[:,1,:], axis=1).max()
        # 0 err_vecs.append([1])

    blured_heatmap = cv2.GaussianBlur(heatmap, (501, 501), 0)
    # byte_heatmap = t = (blured_heatmap * 255.0 / blured_heatmap.max()).astype(np.uint8)
    # plt.figure()
    # plt.imshow(byte_heatmap.T)

    # t = (target * 255.0 / target.max()).astype(np.uint8)


    print( f'total error: {mean_error / len(obj_points)}')



reprojection_error( allCorners, charuco_board.chessboardCorners, rvecs, tvecs, imsize)




def draw_markers( file_name, dict, marker_size, mtx, dist):
    frame = cv2.imread(file_name)
    frame = cv2.undistort(src = frame, cameraMatrix = mtx, distCoeffs = dist)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters =  cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dict,
                                                          parameters=parameters)
    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

    rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size , mtx, dist)
    length_of_axis = 0.1
    imaxis = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    for i in range(len(tvecs)):
        imaxis = cv2.aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
    plt.figure()
    plt.imshow(imaxis)
    plt.grid()
    plt.show()

for img in images:
    draw_markers(img, dict_board, 0.0619, mtx, dist)
    draw_markers(img, dict_marker, 0.20, mtx, dist)


show_undistortion(images[-1], mtx, dist)

# frame = cv2.drawChessboardCorners(frame, (7, 4), res2[1], True)
# frame = cv2.resize(frame, (int(frame.shape[1] / 4), int(frame.shape[0] / 4)))
# frame = cv2.resize(frame, tuple(np.floor_divide(np.array(frame.shape)[::-1], 4)))
# rects = np.floor_divide(np.array(rejectedImgPoints, dtype=int),4)
# cv2.polylines(frame, rects, True, (255, 255, 255), 3)        # for rect inrects:
# cv2.imshow('frame', frame)
# cv2.waitKey(500)
