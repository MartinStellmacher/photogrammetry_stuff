import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import utilities


def detect_corners(images, h_corners, v_corners, square_width, img_scale):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((v_corners * h_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h_corners, 0:v_corners].T.reshape(-1, 2) * square_width
    im_size = None
    result = None
    for fname in images:
        img = utilities.read_scaled_image(fname, img_scale)
        # remember and check the size
        if im_size is None:
            im_size = img.shape[::-1]
        else:
            assert im_size == img.shape[::-1]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (h_corners, v_corners), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_pt_df = pd.DataFrame(corners2.reshape((-1, 2)), columns=['img_pt_x', 'img_pt_y'])
            obj_pt_df = pd.DataFrame(objp, columns=['obj_pt_x', 'obj_pt_y', 'obj_pt_z'])
            idx = pd.MultiIndex.from_product([[fname], img_pt_df.index], names=['image_file', 'point_idx'])
            i_df = pd.concat([img_pt_df, obj_pt_df], axis=1).set_index(idx)
            result = pd.concat([result, i_df])
        else:
            print(f'{fname} failed')
    return pd.Series(im_size, index = ['width', 'height']), result


def calibrate_chessboard( size, points):
    # convert dataframe to cv2 format
    objpoints = []
    imgpoints = []
    image_names = points.index.get_level_values(0).unique()
    for img in image_names:
        img_pts = points.loc[img]
        objpoints.append( img_pts[['obj_pt_x','obj_pt_y','obj_pt_z']].to_numpy())
        imgpoints.append( img_pts[['img_pt_x','img_pt_y']].to_numpy().reshape((-1,1,2)))
    cameraMatrixInit = np.array([[ 1000.,    0., size[0]/2.],
                                 [    0., 1000., size[1]/2.],
                                 [    0.,    0.,           1.]])
    # perform calibration
    rpe, mtx, dist, rvecs, tvecs, stddev_intrinsics, stddev_extrinsics, per_view_error = cv2.calibrateCameraExtended(objpoints, imgpoints, tuple( size), cameraMatrixInit, None, flags = cv2.CALIB_FIX_ASPECT_RATIO)
    # optimize image size
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, tuple(size), 1, tuple(size))
    # collect result
    stddev_extrinsics = stddev_extrinsics.reshape((-1, 2, 3))
    extrinsics = None
    for idx, _ in enumerate(image_names):
        i_df = pd.concat([pd.DataFrame([tvecs[idx].reshape(-1)], columns=['tx', 'ty', 'tz']),
                          pd.DataFrame([rvecs[idx].reshape(-1)], columns=['rx', 'ry', 'rz']),
                          pd.DataFrame([stddev_extrinsics[idx, 1]], columns=['stddev_tx', 'stddev_ty', 'stddev_tz']),
                          pd.DataFrame([stddev_extrinsics[idx, 0]], columns=['stddev_rx', 'stddev_ry', 'stddev_rz']),
                          pd.DataFrame([per_view_error[idx]], columns=['rpe'])], axis=1)
        extrinsics = pd.concat([extrinsics, i_df])
    extrinsics = extrinsics.set_index(image_names)
    intrinsics = pd.DataFrame([np.concatenate([[mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]], dist[0]]), stddev_intrinsics.reshape(-1)[:9]], columns=['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'], index=['value', 'stddev'])
    return intrinsics, extrinsics, pd.DataFrame(newcameramtx, columns=['fx', 'fy', 'c']), pd.Series(roi, index=['x', 'y', 'w', 'h'], name='roi')


def perform_calibration(images, board_width, board_height, square_width, point_store, calibration_store, image_scaling=1.0):
    if point_store.exists():
        size = pd.read_hdf(point_store, 'size')
        points = pd.read_hdf(point_store, 'points')
    else:
        size, points = detect_corners(images, board_width, board_height, square_width, image_scaling)
        size.to_hdf(point_store, 'size', mode='w')
        points.to_hdf(point_store, 'points', mode='a')
    if calibration_store.exists():
        intrinsics = pd.read_hdf(calibration_store, 'intrinsics')
        extrinsics = pd.read_hdf(calibration_store, 'extrinsics')
        newcameramtx = pd.read_hdf(calibration_store, 'newcameramtx')
        roi = pd.read_hdf(calibration_store, 'roi')
    else:
        intrinsics, extrinsics, newcameramtx, roi = calibrate_chessboard(size, points)
        intrinsics.to_hdf(calibration_store, 'intrinsics', mode='w')
        extrinsics.to_hdf(calibration_store, 'extrinsics', mode='a')
        newcameramtx.to_hdf(calibration_store, 'newcameramtx', mode='a')
        roi.to_hdf(calibration_store, 'roi', mode='a')
    return intrinsics, extrinsics, size, points, newcameramtx, roi


def calibration_main(image_path, board_width, board_height, square_width, point_store='', calibration_store='', image_scaling=1.0):
    images = list(image_path.glob('*.jpg'))
    intrinsics, extrinsics, size, points, _, _ = perform_calibration(images, board_width, board_height, square_width, point_store, calibration_store, image_scaling)
    print( utilities.calculate_reprojection_error(intrinsics, extrinsics, points))
    utilities.create_corner_visualization_files(Path('data/basement_1/chessboard_calib/out'), images, board_width, board_height, points, 0.25)


if __name__ == '__main__':
    calibration_main(Path('data/basement_1/images'), 7, 4, 0.0885, Path('data/basement_1/chessboard_calib/corners.h5'), Path('data/basement_1/chessboard_calib/calib.h5'), image_scaling=0.25)



# image undistort
# mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, shape, cv2.CV_32FC1)
# # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# for img_name in images:
#     img = cv2.imread(img_name, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
#     dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#     cv2.imshow('img', dst)
#     if cv2.waitKey(-1) & 0xFF == ord('q'):
#         break


#  Scaling down by 4 helps and all but two images succeed!
#   STL04007.JPG extrem verzerrt in der Ecke aber komplett
#   STL04031.JPG mittig aber fast Bild-füllend
#  Scaling down by 8
#   STL04012.JPG
#   STL04016.JPG


