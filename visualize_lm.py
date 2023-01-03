import cv2
from src.face_detector import FaceDetector
from src.landmark_estimator import LandmarkEstimator

#landmark_index = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
landmark_index = [26, 22, 21, 17, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8]

if __name__ == "__main__":
    video_path = 0
    # model initial
    front_face = FaceDetector('dlib_frontal')
    cnn_face = FaceDetector('dlib_cnn')
    landmark_v1 = LandmarkEstimator('dlib_v1')
    landmark_v2 = LandmarkEstimator('dlib_v2')
    # color map:
    color_map = {
        'bk': (0, 0, 0),
        'front': (255, 0, 0),
        'cnn': (0, 255, 0),
        'r': (0 ,0, 255)
    }
    # inference
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        im2show = frame.copy()
        # frontal face
        front_image_info = front_face.inference(frame)
        front_image_info.box_plot(im2show, color=color_map['bk'])
        landmark_v1.inference(frame, front_image_info, from_cv2=True)
        front_image_info.landmark_plot(im2show, index=None, color=color_map['front'], num=False)
        front_image_info.landmark_plot(im2show, index=landmark_index, color=color_map['front'], num=True)
#        cnn_image_info = cnn_face.inference(frame)
#        cnn_image_info.box_plot(im2show, color=color_map['bk'], show_conf=True)
#        landmark_v2.inference(frame, cnn_image_info, from_cv2=True)
#        cnn_image_info.landmark_plot(im2show, index=None, color=color_map['cnn'])
        cv2.imshow("test", im2show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('s'):
            print("write result image to figures/visualize_lm_result.jpg")
            cv2.imwrite("figures/visualize_lm_result.jpg", im2show)
    cap.release()
    cv2.destroyAllWindows()
