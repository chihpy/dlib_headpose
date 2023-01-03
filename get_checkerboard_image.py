"""
Capture image from webcam with {camera_idx}, press 's' to save image to {save_dir}
note: 
- default frame shape is (480, 640, 3)

Usage - get c170 checkerboard image
  $ python get_checkerboard_image.py --save_dir=./calibration/intrinsic

"""
import argparse
import os
import cv2
from datetime import datetime

def dir_gen(dir):
    if os.path.isdir(dir):
        print(f"{dir} already exist")
    else:
        print("mkdir " + save_dir)
        os.makedirs(save_dir)

def get_camera_name(dir):
    return os.path.basename(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        help="save image directory",
                        default=r"temp\\")
    parser.add_argument("--camera_idx", type=int,
                        help="webcam index",
                        default=0)
    args = parser.parse_args()

    save_dir = args.save_dir
    dir_gen(save_dir)
    camera_idx = args.camera_idx
    camera_name = get_camera_name(save_dir)
    print(f"get image from {camera_name} with index {camera_idx}, user should check camera correct")
    print(f"save image to " + save_dir)
    save_cnt = 0
    frame_cnt = 0
    cap = cv2.VideoCapture(camera_idx)
    # if camera == sj4k: get (1280, 720) image
    if camera_name == 'sj4k':
        WIDTH = 1280
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt == 0:
            print("frame shape: {}".format(frame.shape))
            frame_cnt+=1
        print("press 's' to save image, 'q' to quite", end='\r')
        cv2.imshow("result", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            strtime = datetime.now().strftime('%m%d%H%M%S')
            save_path = os.path.join(save_dir, strtime + f"{save_cnt}.jpg")
            print('save image to: ', end='')
            print(save_path)
            cv2.imwrite(save_path, frame)
            save_cnt+=1

    cap.release()
    cv2.destroyAllWindows()