import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import multiprocessing as mp

import utilities


def create_charucoboard(dict_type, charuco_board_para):
    dict = cv2.aruco.getPredefinedDictionary(dict_type)
    board = cv2.aruco.CharucoBoard_create(*charuco_board_para, dict)
    return dict, board

def detect_charucoboard(fname, img_scale, charuco_board_para, dict_type):
    dict, board = create_charucoboard(dict_type, charuco_board_para)
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


def detect_charucoboards(images, charuco_board_para, dict_type, image_scaling=1.0):
    result = None
    im_size = None
    pool = mp.Pool(max(1, mp.cpu_count()-2))
    results = [pool.apply_async(detect_charucoboard, args=(fname, image_scaling, charuco_board_para, dict_type)) for fname in images]
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


def calibrate_charuco(points, imsize, board):
    # fx == fy, absolute value has no meaning if CALIB_FIX_ASPECT_RATIO is specified
    # assume perfectly centered optics for the start
    cameraMatrixInit = np.array([[ 1000.,    0., imsize.width/2.],
                                 [    0., 1000., imsize.height/2.],
                                 [    0.,    0.,           1.]])
    img_points = []
    img_ids = []
    image_names = points.index.get_level_values(0).unique()
    for img in image_names:
        img_pts = points.loc[img]
        img_ids.append(img_pts.index.to_numpy().reshape((-1,1)))
        img_points.append( img_pts[['img_pt_x','img_pt_y']].to_numpy().reshape((-1,1,2)))
    distCoeffsInit = np.zeros((5,1))
    # flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    flags = (cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=img_points,
                      charucoIds=img_ids,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print(f'optics decentered {(np.divide(camera_matrix[0:2,2],np.array(imsize)/2)-1)*100} [%]')
    print(f'optics decentered {camera_matrix[0:2,2] - np.array(imsize)/2} [Pixel]')

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, tuple(imsize), 1,
                                                      tuple(imsize))
    stddev_extrinsics = stdDeviationsExtrinsics.reshape((-1, 2, 3))
    extrinsics = None
    for idx, _ in enumerate(image_names):
        i_df = pd.concat([pd.DataFrame([translation_vectors[idx].reshape(-1)], columns=['tx', 'ty', 'tz']),
                          pd.DataFrame([rotation_vectors[idx].reshape(-1)], columns=['rx', 'ry', 'rz']),
                          pd.DataFrame([stddev_extrinsics[idx, 1]], columns=['stddev_tx', 'stddev_ty', 'stddev_tz']),
                          pd.DataFrame([stddev_extrinsics[idx, 0]], columns=['stddev_rx', 'stddev_ry', 'stddev_rz']),
                          pd.DataFrame([perViewErrors[idx]], columns=['rpe'])], axis=1)
        extrinsics = pd.concat([extrinsics, i_df])
    extrinsics = extrinsics.set_index(image_names)
    intrinsics = pd.DataFrame([[camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]]+distortion_coefficients.reshape((-1)).tolist(), stdDeviationsIntrinsics.reshape(-1)[:9]], columns=['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'], index=['value', 'stddev'])
    return intrinsics, extrinsics, pd.DataFrame(newcameramtx, columns=['fx', 'fy', 'c']), pd.Series(roi, index=['x', 'y', 'w', 'h'], name='roi')


def perform_calibration( images, board_width, board_height, square_width, marker_width, dict_type, point_store, calibration_store, image_scaling=1.0):
    charuco_board_para = (board_width, board_height, square_width, marker_width)
    if point_store.exists():
        size = pd.read_hdf(point_store, 'size')
        points = pd.read_hdf(point_store, 'points')
    else:
        size, points = detect_charucoboards(images, charuco_board_para, dict_type, image_scaling)
        size.to_hdf(point_store, 'size', mode='w')
        points.to_hdf(point_store, 'points', mode='a')
    if calibration_store.exists():
        intrinsics = pd.read_hdf(calibration_store, 'intrinsics')
        extrinsics = pd.read_hdf(calibration_store, 'extrinsics')
        newcameramtx = pd.read_hdf(calibration_store, 'newcameramtx')
        roi = pd.read_hdf(calibration_store, 'roi')
    else:
        _, board = create_charucoboard(dict_type, charuco_board_para)
        intrinsics, extrinsics, newcameramtx, roi = calibrate_charuco(points, size, board)
        intrinsics.to_hdf(calibration_store, 'intrinsics', mode='w')
        extrinsics.to_hdf(calibration_store, 'extrinsics', mode='a')
        newcameramtx.to_hdf(calibration_store, 'newcameramtx', mode='a')
        roi.to_hdf(calibration_store, 'roi', mode='a')
    return intrinsics, extrinsics, size, points, newcameramtx, roi


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




