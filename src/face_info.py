"""
TODO: 
- [x] hp_axis_viz in pixel coord. [ ] debug
- [x] hp_box_viz in pixel coord.: [-] fail
- [ ] reimplement FaceInfo and ImageInfo
- [ ] dump ImageInfo
- [ ] print ImageInfo
"""
import cv2
import numpy as np
from src.face_3d import get_cube_pts

class FaceInfo:
    def __init__(self, bndbox):
        self.bndbox = bndbox  # single float value
        self.confidence = None  # list of int in [xmin, ymin, xmax, ymax] order
        self.landmarks = None
        self.landmark_model_name = None
        # head pose info
        self.r_vec = None
        self.t_vec = None
        self.r_mat = None
        self.world_to_cam = None  # 4x4 matrix
        self.euler_angle = None  # pitch, yaw, roll
        # animation in camera coord.
        self.landmarks_in_cam = None
    
    def get_2d_center(self):
        if self.landmarks is None:
            raise("landmarks is None")
        landmarks = self.landmarks  # list of tuple
        ct_x = 0
        ct_y = 0
        cnt = 0
        for pt in landmarks:
            cnt+=1
            ct_x += pt[0]
            ct_y += pt[1]
        return int(ct_x/cnt), int(ct_y/cnt)

class ImageInfo:
    def __init__(self, image_size, num_landmark=68):
        self.faces = []
        self.dlib_rects = None
        self.height = image_size[0]
        self.width = image_size[1]
        self.num_landmark = num_landmark
        self.cam_mat = None
        self.dist_coeffs = None

    
    def get_num_face(self):
        return len(self.faces)

    
    def box_plot(self, im2show, color=(255, 255, 255), show_conf=False):
        
        if self.get_num_face() == 0:
            return
        else:
            for face in self.faces:
                xmin, ymin, xmax, ymax = face.bndbox
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                cv2.rectangle(im2show, (xmin, ymin), (xmax, ymax), color, 3)
                if show_conf:
                    conf = face.confidence
                    if conf is None:
                        conf = -1
                    txt_info = "{:.2f}".format(conf)
                    ((text_width, text_height), _) = cv2.getTextSize(txt_info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    cv2.putText(
                        im2show,
                        text=txt_info,
                        org=(xmin, ymin - int(0.3 * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.35, 
                        color=color, 
                        lineType=cv2.LINE_AA,
                    )
    def landmark_plot(self, im2show, index=None, color=(255, 255, 255), num=False):
        if index is None:
            index = [i for i in range(self.num_landmark)]
        for face in self.faces:
            landmarks = face.landmarks
            if landmarks is None:
                continue
            for cnt, idx in enumerate(index):
                pt = landmarks[idx]
                cv2.circle(im2show, (pt[0], pt[1]), 3, color, -1)
                if num:
                    txt_info = str(cnt)
                    ((text_width, text_height), _) = cv2.getTextSize(txt_info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    cv2.putText(
                    im2show,
                    text=txt_info,
                    org=(pt[0], pt[1] - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,  
                    color=(0, 0, 0), 
                    lineType=cv2.LINE_AA,
                    )
    
    def hp_axis_plot(self, im2show):
        """
        x-axis-blue
        y-axis-green
        z-axis-red
        """
        axisLength = 100  # change s.t. dependent on bounding box
        world_ptx = np.array([[1], [0], [0]])
        world_pty = np.array([[0], [1], [0]])
        world_ptz = np.array([[0], [0], [1]])
        for face in self.faces:
            ct_x, ct_y = face.get_2d_center()
            r_mat = face.r_mat
            cam_ptx = np.dot(r_mat, world_ptx)
            cam_pty = np.dot(r_mat, world_pty)
            cam_ptz = np.dot(r_mat, world_ptz)

            cv2.arrowedLine(im2show, (ct_x, ct_y), (ct_x + int(axisLength * cam_ptx[0][0]), ct_y + int(axisLength * cam_ptx[1][0])), (255, 0, 0), thickness=2)
            cv2.arrowedLine(im2show, (ct_x, ct_y), (ct_x + int(axisLength * cam_pty[0][0]), ct_y + int(axisLength * cam_pty[1][0])), (0, 255, 0), thickness=2)
            cv2.arrowedLine(im2show, (ct_x, ct_y), (ct_x + int(axisLength * cam_ptz[0][0]), ct_y + int(axisLength * cam_ptz[1][0])), (0, 0, 255), thickness=2)
    
    def hp_cube_plot(self, im2show):
        """
        cube in world coord. check visualize_face_in_world
        """
        # get cube corner
        xmin, xmax, ymin, ymax, zmin, zmax = get_cube_pts()
        cube_corner = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            #
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax]
        ])
        for face in self.faces:
            # project corner to pixel coord
            reprojectdst, _ = cv2.projectPoints(cube_corner,
                                                face.r_vec,
                                                face.t_vec,
                                                self.cam_mat,
                                                self.dist_coeffs)
            reprojectdst = np.squeeze(reprojectdst)
            line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]
            for start, end in line_pairs:
                cv2.line(im2show, (int(reprojectdst[start][0]), int(reprojectdst[start][1])), (int(reprojectdst[end][0]), int(reprojectdst[end][1])), (0, 0, 255))

            


