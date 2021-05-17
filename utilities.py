import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_scaled_image(fname, img_scale=1):
    # read all images in the same orientation
    img = cv2.imread(fname, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
    # downscale because findChessboardCorners can't handle multi megapixel images :(
    if img_scale != 1:
        img = cv2.resize(img, (int(img.shape[1] * img_scale), int(img.shape[0] * img_scale)))
    return img


def intrinsics_to_mtx_and_dist( intrinsics):
    mtx = np.array([[intrinsics.fx.value, 0., intrinsics.cx.value],
                    [0., intrinsics.fy.value, intrinsics.cy.value],
                    [0., 0., 1.]])
    dist = np.array([intrinsics.k1.value, intrinsics.k2.value, intrinsics.p1.value, intrinsics.p2.value, intrinsics.k3.value])
    return mtx, dist


def calculate_reprojection_error(intrinsics, extrinsics, points):
    mean_error = 0
    mtx, dist = intrinsics_to_mtx_and_dist(intrinsics)
    for index, row in extrinsics.iterrows():
        imgpoints2, _ = cv2.projectPoints(points.loc[index][['obj_pt_x', 'obj_pt_y', 'obj_pt_z']].to_numpy(), row[['rx','ry','rz']].to_numpy(), row[['tx','ty','tz']].to_numpy(), mtx, dist)
        # don't use NORM_L2 here, because it takes the root which we would square next ...
        mean_error +=  np.sum(np.sum(np.square(points.loc[index][['img_pt_x', 'img_pt_y']].to_numpy() - imgpoints2.reshape((-1,2))), axis=1))/len(imgpoints2)
    # the result is the same as Reprojectionerror reported by cv2.calibrateCamera(): (np.power(extrinsics.rpe,2).sum()/len(extrinsics))**0.5
    return (mean_error/len(extrinsics))**0.5


def create_corner_visualization_cv2(images, board_width, board_height, points, image_scaling=1.0):
    for image in images:
        img = read_scaled_image(image, image_scaling)
        corner_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        try:
            corners2 = points.xs(image).loc[:,('img_pt_x', 'img_pt_y')].to_numpy().reshape((-1,1,2))
            corner_img = cv2.drawChessboardCorners(corner_img, (board_width, board_height), corners2, True)
        except:
            corner_img = cv2.rectangle(corner_img, (0, 0), (corner_img.shape[1]-1, corner_img.shape[0]-1), (0, 0, 255), 5)
        cv2.imshow('markers', corner_img)
        cv2.waitKey(-1)


def create_corner_visualization_plt(images, board_width, board_height, points, image_scaling=1.0):
    square_dim = math.ceil(len(images) ** 0.5)
    fig, axs = plt.subplots(square_dim, square_dim)  # len(images))
    fig.suptitle('Detected Chessboard Corners')
    for x in range(square_dim):
        for y in range(square_dim):
            axs[y, x].get_xaxis().set_visible(False)
            axs[y, x].get_yaxis().set_visible(False)
    for i, image in enumerate(images):
        img = read_scaled_image(image, image_scaling)
        corner_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        try:
            corners2 = points.xs(image).loc[:,('img_pt_x', 'img_pt_y')].to_numpy().reshape((-1,1,2))
            corner_img = cv2.drawChessboardCorners(corner_img, (board_width, board_height), corners2, True)
        except:
            corner_img = cv2.rectangle(corner_img, (0, 0), (corner_img.shape[1]-1, corner_img.shape[0]-1), (255, 0, 0), 25)
        ax = axs[math.floor(i / square_dim), math.floor(i % square_dim)]
        ax.imshow(corner_img)
        ax.set_title(os.path.basename(image))
    plt.show()

