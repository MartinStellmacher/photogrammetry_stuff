# Setup
## Camera
Zeiss Batis 18mm, focus fixed at 2m, 24*36 mm², 7952x5304
## Aruco Marker
cv2.aruco.DICT_6X6_250, IDs 40-48
## Charuco Board
    cv2.aruco.DICT_4X4_50
    board size [fields] 5x8
    IDs 0-19
    marker size [pixel] 731 [m] 0.0619
    field size [pixel] 1045 [m] 0.0885
    board size [pixel] 5906 x 8858 [m] 0.50 x 0.75 ["] 19.68504 x 29.52756 (at 300DPI  5905,5 x 8858,3 Pixel)
    measured in PhotoShop:
        long (8608-250)/300*2,54/8        1044,75 => 88,4555 mm
        short (5564-341)/300*2,54/5        1044,6 => 88,4428 mm
##Experiments
###Aruco
ToDo: Marker need a white boarder of at least the same width as the black boarder !!!

### Chessboard
Only possible for 1/16th of image size because otherwise the checkerboard isn't detected robustly....

Best overall result: calibrate 0.44, total 0.072

#### 1/4 size
Reprojectionerror reported by cv2.calibrateCamera(): 0.4484306472864837

    (1988, 1326)
    [[1.02849864e+03 0.00000000e+00 9.92687906e+02]
     [0.00000000e+00 1.02849864e+03 6.67820589e+02]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    [[-8.26394354e-02  5.43711891e-02 -6.36383960e-04 -7.83326198e-05
      -6.84988097e-03]]


### Charuco
#### Charuco 1/4 size
    (1988, 1326)
    0.1544299370880001
    [[1.03185816e+03 0.00000000e+00 9.93933319e+02]
     [0.00000000e+00 1.03185816e+03 6.66786321e+02]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    [-0.08418540666508477, 0.0597541068351983, -0.000684028911722229, 2.47930280277697e-05, -0.01016865512827588]

#### Charuco full size
    (7952, 5304)
    0.5912173034174356
    [[4.12269741e+03 0.00000000e+00 3.97812095e+03]
     [0.00000000e+00 4.12269741e+03 2.67010659e+03]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    [-0.08592457028182002, 0.062382333173428714, -0.0005353250409795117, 6.457533439894114e-05, -0.012035997908285882]


### from the documentation
#### calibrateCamera
    Returns the overall RMS re-projection error.
    Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error, that is, the total sum
    of squared distances between the observed feature points imagePoints and the projected (using the current estimates
    for camera parameters and the poses) object points objectPoints. See projectPoints for details.

#### projectPoints
    Opposite of undistortPoints()
    world -> image requires rvec, tvec, matrix, distortion



