import numpy as np
import cv2
import glob
import os
import pandas as pd


def detect_corners(images, h_corners, v_corners):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((v_corners * h_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h_corners, 0:v_corners].T.reshape(-1, 2)
    im_size = None
    result = None

    for fname in images:
        # read all images in the same orientation
        img = cv2.imread(fname, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
        # downscale because findChessboardCorners can't handle multi megapixel images :(
        img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
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
            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (h_corners, v_corners), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            # combine result
            img_pt_df = pd.DataFrame(corners2.reshape((-1, 2)), columns=['img_pt_x', 'img_pt_y'])
            obj_pt_df = pd.DataFrame(objp, columns=['obj_pt_x', 'obj_pt_y', 'obj_pt_z'])
            idx = pd.MultiIndex.from_product([[fname], img_pt_df.index], names=['image_file', 'point_idx'])
            i_df = pd.concat([img_pt_df, obj_pt_df], axis=1).set_index(idx)
            result = pd.concat([result, i_df])
            # img_name_s = pd.Series([fname]*len(objp), name='img_name')
            # i_df = pd.concat([img_name_s, img_pt_df, obj_pt_df], axis=1)
            # result = pd.concat([result, i_df], ignore_index=True)
        else:
            print(f'{fname} failed')
    cv2.destroyAllWindows()
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
    rpe, mtx, dist, rvecs, tvecs, stddev_intrinsics, stddev_extrinsics, per_view_error = cv2.calibrateCamera(objpoints, imgpoints, tuple( size), cameraMatrixInit, None, flags = cv2.CALIB_FIX_ASPECT_RATIO)
    # optimize image size
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, tuple(size), 1, tuple(size))
    # collect result
    # stddev_intrinsics (18, 1) fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy
    # stddev_extrinsics (360, 1) R0,T0,…,RM−1,TM−1
    vecs = None
    for idx, _ in enumerate(image_names):
        i_df = pd.concat([pd.DataFrame([tvecs[idx].reshape(-1)], columns=['tx', 'ty', 'tz']),
                          pd.DataFrame([rvecs[idx].reshape(-1)], columns=['rx', 'ry', 'rz']),
                          pd.DataFrame([per_view_error[idx]], columns=['rpe'])], axis=1)
        vecs = pd.concat([vecs, i_df])
    vecs = vecs.set_index(image_names)
    intrinsics = pd.DataFrame([np.concatenate([[mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]], dist[0]]), stddev_intrinsics.reshape(-1)[:9]], columns=['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'], index=['value', 'std_dev'])

    return rpe, pd.DataFrame(mtx, columns=['fx', 'fy', 'c']), pd.Series(dist[0], name='distortion'), vecs,\
           pd.DataFrame(newcameramtx, columns=['fx', 'fy', 'c']), pd.Series(roi, index=['x', 'y', 'w', 'h'], name='roi')

point_store = 'data/basement_1/chessboard_calib/corners.h5'
if os.path.exists(point_store):
    size = pd.read_hdf(point_store, 'size')
    points = pd.read_hdf(point_store, 'points')
else:
    size, points = detect_corners(glob.glob(r"data/basement_1/images/*.JPG"), 7, 4)
    size.to_hdf(point_store, 'size')
    points.to_hdf(point_store, 'points')

rpe, mtx, dist, vecs, newcameramtx, roi = calibrate_chessboard(size, points)

# calibration_store = 'data/basement_1/chessboard_calib/calib.h5'
# if os.path.exists(calibration_store):
#     with open('data/basement_1/chessboard_calib/chessboard_calib.pkl', 'rb') as f:
#         rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi = pickle.load(f)
# else:
#     rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi = calibrate_chessboard(size, points)
#     with open('data/basement_1/chessboard_calib/chessboard_calib.pkl', 'wb') as f:
#         pickle.dump((rpe, mtx, dist, rvecs, tvecs, newcameramtx, roi), f)

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


