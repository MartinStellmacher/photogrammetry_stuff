import numpy as np
import cv2
import time
from pathlib import Path
import pickle
import pandas as pd
import multiprocessing as mp
import sys

import utilities


def read_charucoboard(fname, img_scale, charuco_board_para, dict_type):
    dict = cv2.aruco.getPredefinedDictionary(dict_type)
    board = cv2.aruco.CharucoBoard_create(*charuco_board_para, dict)
    frame = utilities.read_scaled_image(fname, img_scale=img_scale)
    im_size = frame.shape[::-1]
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dict)
    if len(corners) > 0:
        n_final, pts_final, ids_final = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
        if pts_final is not None and ids_final is not None and n_final > 3:
            img_pt_df = pd.DataFrame(pts_final.reshape((-1, 2)), columns=['img_pt_x', 'img_pt_y'],
                                     index=pd.MultiIndex.from_product([[fname], ids_final.reshape(-1)],
                                                                      names=['image_file', 'point_id']))
        return fname, im_size, img_pt_df
    else:
        return fname, im_size, None


def read_charucoboards(images, charuco_board_para, dict_type, image_scaling=1.0):
    result = None
    im_size = None
    pool = mp.Pool(max(1, mp.cpu_count()-2))
    results = [pool.apply_async(read_charucoboard, args=(fname, image_scaling, charuco_board_para, dict_type)) for fname in images]
    for r in results:
        fname, one_size, res = r.get()
        if im_size is None:
            im_size = one_size
        else:
            assert im_size == one_size
        if res is not None:
            result = pd.concat([result, res])
        else:
            print(f'{fname} failed')
    return pd.Series(im_size, index = ['width', 'height']), result


def read_charucoboards_seq(images, charuco_board_para, dict_type, image_scaling=1.0):
    dict = cv2.aruco.getPredefinedDictionary(dict_type)
    board = cv2.aruco.CharucoBoard_create(*charuco_board_para, dict)
    result = None
    im_size = None
    for img in images:
        print("=> Processing image {0}".format(img))
        frame = utilities.read_scaled_image(img, img_scale=image_scaling)
        # check size
        if im_size is None:
            im_size = frame.shape[::-1]
        else:
            assert im_size == frame.shape[::-1]
        # first detect the markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dict)
        # then find the chessboard corners
        if len(corners)>0:
            n_final, pts_final, ids_final = cv2.aruco.interpolateCornersCharuco(corners,ids,frame,board)
            if pts_final is not None and ids_final is not None and n_final>3:
                img_pt_df = pd.DataFrame(pts_final.reshape((-1, 2)), columns=['img_pt_x', 'img_pt_y'],
                             index=pd.MultiIndex.from_product([[img], ids_final.reshape(-1)],
                                                              names=['image_file', 'point_id']))
                result = pd.concat([result, img_pt_df])
                if len( board.chessboardCorners) != n_final:
                    print(f'\tdetected only {n_final} of {board.chessboardCorners} chessboard corners')
            else:
                print('\tfailed')
    return pd.Series(im_size, index = ['width', 'height']), result


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


def perform_calibration( images, board_width, board_height, square_width, marker_width, dict_type, point_store, calibration_store, image_scaling=1.0):
    charuco_board_para = (board_width, board_height, square_width, marker_width)
    if point_store.exists():
        with point_store.open('rb') as f:
            allCorners, allIds, imsize = pickle.load( f)
    else:
        t0 = time.time()
        im_size, corners = read_charucoboards(images, charuco_board_para, dict_type, image_scaling)
        print(f'{time.time()-t0:0.1f}s')
        sys.exit()
        # with point_store.open('wb') as f:
        #     pickle.dump((allCorners, allIds, imsize), f)

    # if os.path.exists( 'charuco_calibration.pkl'):
    #     with open('charuco_calibration.pkl', 'rb') as f:
    #         ret, mtx, dist, rvecs, tvecs = pickle.load( f)
    # else:
    #     ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize, charuco_board)
    #     with open('charuco_calibration.pkl', 'wb') as f:
    #         pickle.dump( (ret, mtx, dist, rvecs, tvecs), f)


def calibration_main(image_path, board_width, board_height, square_width, marker_width, dict_type, point_store='',
                     calibration_store='', image_scaling=1.0):
    images = list(image_path.glob('*.jpg'))
    intrinsics, extrinsics, size, points, _, _ = perform_calibration(images, board_width, board_height, square_width,
                                                                     marker_width, dict_type, point_store,
                                                                     calibration_store, image_scaling)
    print( utilities.calculate_reprojection_error(intrinsics, extrinsics, points))
    utilities.create_corner_visualization_files(Path('data/basement_1/chessboard_calib/out'), images, board_width,
                                                board_height, points, 0.25)


# Markergröße Pixel: 731 [m]: 0.0619
# Kachelgröße Pixel: 1045 [m]: 0.0885
if __name__ == '__main__':
    calibration_main(Path('data/basement_1/images'), 5, 8, 0.0885, 0.0619,
                     cv2.aruco.DICT_4X4_50,
                     Path('data/basement_1/charuco_calib/corners.h5'),
                     Path('data/basement_1/charuco_calib/calib.h5'), image_scaling=0.25)



# print(imsize)
# print(np.array(imsize)/2)
# print(ret)
# print(mtx)
# print(dist.ravel().tolist())


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


    # print( f'total error: {mean_error / len(obj_points)}')



# reprojection_error( allCorners, charuco_board.chessboardCorners, rvecs, tvecs, imsize)




