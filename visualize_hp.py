"""
TODO: add intrinsic calibration
"""
import cv2
from src.face_detector import FaceDetector
from src.landmark_estimator import LandmarkEstimator
from src.headpose_estimator import PnpHeadPose

#FD_NAME = 'dlib_frontal'
FD_NAME = 'dlib_cnn'
if FD_NAME == 'dlib_cnn':
    LM_NAME = 'dlib_v2'
elif FD_NAME == 'dlib_frontal':
    LM_NAME = 'dlib_v1'
else:
    raise("unknown FD_NAME: "+ FD_NAME)

landmark_index = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]  # for visualize only: highlight selected pts

if __name__ == "__main__":
    video_path = 0
    # model initial
    fd = FaceDetector(FD_NAME)
    lm = LandmarkEstimator(LM_NAME)
    hp = PnpHeadPose()
    # color map:
    color_map = {
        'bk': (0, 0, 0),
        'r': (0, 0, 255),
        'g': (0 ,255, 0),
        'b': (0, 0, 255)
    }
    # inference
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        im2show = frame.copy()
        # frontal face
        image_info = fd.inference(frame)
        image_info.box_plot(im2show, color=color_map['bk'])
        lm.inference(frame, image_info, from_cv2=True)
        image_info.landmark_plot(im2show, index=None, color=color_map['bk'])
        image_info.landmark_plot(im2show, index=landmark_index, color=color_map['b'])
        hp.inference(image_info, print_euler=True)
        image_info.hp_axis_plot(im2show)
        image_info.hp_cube_plot(im2show)
        cv2.imshow("test", im2show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
