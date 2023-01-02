# Head Pose estimation via dlib models and solvepnp
# Demo
## Face Detection:
- compare hog+svm(blue) and cnn(green)
```
python viaualize_fd.py
```
## Landmark Estimation
- compare hog+landmark_v1(blue) and cnn+landmark_v2(green)
```
python visualize_lm.py
```
## Canonical Face 3d
- visualize Canonical Face Mdoel in World Coord.
```
python visualize_face_in_world.py
```
  - 58 pts only
# TODO:
- [x] dlib_frontal_face_detection(HOG) + dlib_landmark(landmark_v1)
- [x] dlib_cnn_face_detection(CNN) + dlib_landmark(landmark_v2)
- [x] visualize canonical face model in world coordinate system
- [x] head pose estimation with solvePnP
  - [x] axis visualization
  - [x] cube visualization
- [ ] visualize canonical face model in camera coordinate system
- [ ] camera intrinsic calibration
# Models
- Dlib mmod cnn based face detector [cnn_face_detector](https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2)
  - note: This model is slow using CPU but mobile gpu like NVIDIA GeForce MX250 run smooth enough in webcam
- The well trained dlib landmark 68 model cam be downloaded from [landamrk_v1](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
  - regression tree
- Robust landmark model use cnn based face detector for large head pose variation: [landmark_v2](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2)
  - regression tree

# Reference
- [dlib-project](https://github.com/davisking/dlib)
- [dlib-models](https://github.com/davisking/dlib-models)
  - landmark_v1: shape_predictor_68_face_landmarks.dat.bz2
  - landmark_v2: shape_predictor_68_face_landmarks_GTX.dat.bz2
  - cnn_face_detector: mmod_human_face_detector.dat.bz2
- [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)
  - python dlib get headpose example

## Reference Papers
- [dlib_cnn_face_detector](https://arxiv.org/abs/1502.00046)
  - paper: Max-Margin Object Detection(mmod)
- [dlib_landmark_v1](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf)
  - paper: One Millisecond Face Alignment with an Ensemble of Regression Trees
  - dlib hog face detector only
- [dlib_landmark_v2](https://gitlab.com/visualhealth/vhpapers/real-time-facealignment)
  - paper: Real-time face alignment: evaluation methods, training strategies and implementation optimization
  - dlib CNN/hog face detector