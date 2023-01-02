import cv2
from src.face_detector import FaceDetector

if __name__ == "__main__":
    video_path = 0
    # model initial
    front_face = FaceDetector('dlib_frontal')
    cnn_face = FaceDetector('dlib_cnn')
    # color map:
    color_map = {
        'front': (255, 0, 0),
        'cnn': (0, 255, 0)
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
        front_image_info.box_plot(im2show, color=color_map['front'])
        # cnn face
        cnn_image_info = cnn_face.inference(frame)
        cnn_image_info.box_plot(im2show, color=color_map['cnn'], show_conf=True)
        cv2.imshow("test", im2show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
