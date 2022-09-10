# Head Pose estimation via dlib models and solvepnp
The well trained dlib landmark 68 model cam be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Demo
```
python demo_headpose_webcam.py
```
# TODO:
- [x] apply dlib_face_detection + dlib_landmark + cv2_solvepnp on webcam demo
- [ ] dlib landmark v2
- [ ] another face detection model
- [ ] evaluation
# Reference
- [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)
  - python dlib get headpose example