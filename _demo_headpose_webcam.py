import cv2
from dlib_frontal_face_detection import Face_detection
from dlib_landmark_68 import Landmark_detection_68
from pnp_headpose import get_head_pose

LANDMARK_DAT = 'models/shape_predictor_68_face_landmarks.dat'

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# model initial
face_detection = Face_detection()
landmark_detection_68 = Landmark_detection_68(LANDMARK_DAT)

cap = cv2.VideoCapture(0)
#if not cap.isOpened():
#    print("Unable to connect to camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Unable to connect to camera.")
        break
    im2show = frame.copy()
    #face_rects = detector(frame, 0)
    face_rects, face_boxes = face_detection.inference(frame)
    face_detection.draw_face(im2show, face_boxes)
    for rect in face_rects:
        pts = landmark_detection_68.inference(frame, face_rects[0])
        landmark_detection_68.draw_pts(im2show, pts)
        reprojectdst, euler_angle = get_head_pose(pts)
        for start, end in line_pairs:
            cv2.line(im2show, (int(reprojectdst[start][0]), int(reprojectdst[start][1])), (int(reprojectdst[end][0]), int(reprojectdst[end][1])), (0, 0, 255))
    
    cv2.imshow('demo', im2show)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()