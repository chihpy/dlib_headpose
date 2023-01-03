"""
given {image_dir}: contain bag of checkerboard image
apply OpenCV's calibration procedure to get camera intrinsic params
include camera matrix and distoration coefficient
write result json file to {save_dir}

Usage - 
  $ python intrinsic_calibrate.py --jsonpath=./calibration/camera.json --image_dir=/home/pymi/dlib_project/dlib_headpose/calibration/intrinsic/ --viz

# reference:
1. checkerboard image: https://markhedleyjones.com/projects/calibration-checkerboard-collection
2. OpenCV's calibration: https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html
"""
import argparse
import os
import glob
import numpy as np
import cv2
import json
# import yaml  # source code from "efficiency gaze" using yaml format to save intrinsic param.

def dir_gen(dir):
    if os.path.isdir(dir):
        print(f"{dir} already exist")
    else:
        print("mkdir " + dir)
        os.makedirs(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonpath", type=str,
                        help="json file include intrinsic param savepath",
                        default="temp\\camera.json")
    parser.add_argument("--image_dir", type=str,
                        help="directory contain checkerboard image",
                        default="temp\\")
    parser.add_argument("--viz", action='store_true',
                        help='visualize checkerboard detection result')
    
    args = parser.parse_args()
    # setup: path
    image_dir = args.image_dir
    json_path = args.jsonpath
    print("read image from " + image_dir)
    print("write camera param. to " + json_path)
    # steup: general
    VIZ = args.viz
    if VIZ:
        print("visualize checkerboad detection result")
    dir_gen(os.path.dirname(json_path))
    # setup: reference coordinate system (objects pts)
    ## stop the iteration when specified accuracy, epsilon, is reached or
    ## specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 3D points real world coordinates
    # note: checkerboard has size: 7x10
    CHECKERBOARD = (7, 10)
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                          * CHECKERBOARD[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                   0:CHECKERBOARD[1]].T.reshape(-1, 2)
    print('object_3d shape: {}'.format(objectp3d.shape))
    print(objectp3d[:, :10, :])
    # Vector for 3D points
    threedpoints = []
    # Vector for 2D points
    twodpoints = []
#    if os.path.isdir(image_dir):
#        print(image_dir + " is dir")
#    else:
#        print(image_dir + "not dir")
#    image_names = os.listdir(image_dir)
#    print("{} name found".format(len(image_names)))
    images = glob.glob(image_dir + r'*.jpg')
    print(images)
    print("num_images found: {}".format(len(images)))
    if len(images) == 0:
        raise ValueError("no image found in " + image_dir)
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        if VIZ:
            # plot corner and imshow
            im2show = image.copy()
            if ret:
                for cnt, pt in enumerate(corners):
                    ptx = int(pt[0, 0])
                    pty = int(pt[0, 1])
                    cv2.circle(im2show, (ptx, pty), 3, (0, 255, 0), -1)
                    txt_info = str(cnt)
                    ((text_width, text_height), _) = cv2.getTextSize(txt_info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    cv2.putText(
                    im2show,
                    text=txt_info,
                    org=(ptx, pty - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,  
                    color=(0, 0, 0), 
                    lineType=cv2.LINE_AA,
                    )
                cv2.imshow("detected_corner", im2show)
            else:
                print("corner undetect")
                cv2.imshow("detected_corner", im2show)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret:
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
            threedpoints.append(objectp3d)
            twodpoints.append(corners2)

            if VIZ:
                for pt in corners2:
                    ptx = int(pt[0, 0])
                    pty = int(pt[0, 1])
                    cv2.circle(im2show, (ptx, pty), 3, (255, 0, 0), -1)
                cv2.imshow("refined_corner", im2show)
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image,
                                                  CHECKERBOARD,
                                                  corners2, ret)
                cv2.imshow("cv2.drawChessboard", image)
                cv2.waitKey(0)
        else:
            print("corner on found: {}".format(filename))
    cv2.destroyAllWindows()
    print("accepted num_images: {}".format(len(threedpoints)))

    # calibration here
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # Displaying required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(type(r_vecs))

    print("total number of calibrate image: {}".format(len(r_vecs)))
    print(r_vecs[0].shape)
    #print(r_vecs)

    print("\n Translation Vectors:")
    print(len(t_vecs))
    print(t_vecs[0].shape)

    # write result to file
    calibration = {}
    calibration['camera'] = matrix.tolist()
    calibration['distortion'] = distortion.tolist()
    json_object = json.dumps(calibration, indent=4)
    with open(os.path.join(json_path), 'w') as f:
        f.write(json_object)

#### [deprecation] write result to yaml
#data = {
#    'rms': np.asarray(ret).tolist(),
#    'camera_matrix': np.asarray(matrix).tolist(),
#    'dist_coeff': np.asarray(distortion).tolist()
#}
#with open(os.path.join(save_dir, cam_name+'.yaml'), "w") as f:
#    yaml.dump(data, f)