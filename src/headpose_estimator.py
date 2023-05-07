"""
6DOF head pose estimator via cv2.solvePnP
"""
import numpy as np
import cv2
from src.face_3d import get_face3d_14, get_face3d

####
import json
def get_intrinsic_param(json_path):
    with open(json_path, 'r') as f:
        cam_dict = json.load(f)
    cam_mat = np.array(cam_dict['camera'])
    dist_coeffs = np.array(cam_dict['distortion'])
    return cam_mat, dist_coeffs

def rodrigues(r_in):
    """
    convert rotation matrix to vector or
    convert rotation vector to matrix
    """
    r_out, _ = cv2.Rodrigues(r_in)
    return r_out

def get_trans(r_mat, t_vec):
    """
    Args:
        r_mat: 3x3 ndarray
        t_vec: 3x1 ndarray
    Return:
        transform_mat: 4x4 ndarray
    """
    t_mat = np.zeros((4, 4))
    t_mat[:3, :3] = r_mat
    t_mat[:3, 3] = np.squeeze(t_vec)
    t_mat[3, 3] = 1
    return t_mat

def rmat_to_euler(r_mat):
    """
    pitch, yaw, roll
    """
    temp = np.array([[0], [0], [0]]).astype(np.float64)
    pose_mat = cv2.hconcat((r_mat, temp))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return np.squeeze(euler_angle)
####
class PnpHeadPose:
    def __init__(self,
                 intrinsic_path=None,
                 face_3d=None):
        if intrinsic_path is None:
            cam_mat = np.array([
                [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002],
                [0.0, 6.5308391993466671e+002, 2.3950000000000000e+002],
                [0.0, 0.0, 1.0]])
            dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000])
        else:
            cam_mat, dist_coeffs = get_intrinsic_param(intrinsic_path)
        self.cam_mat = cam_mat
        self.dist_coeffs = dist_coeffs

        if face_3d is None:
            #self.landmark_idx = [26, 22, 21, 17, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8]  # landmark 2d index
            self.landmark_idx = [26, 22, 21, 17, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8] # landmark 2d index
            self.object_pts = get_face3d_14()

    def _preprocess(self, face):
        """
        get corresponding 2d landmark
        """
        landmarks = face.landmarks
        selected_landmarks = []
        for index in self.landmark_idx:
            selected_landmarks.append(landmarks[index])
        return np.array(selected_landmarks).astype(np.float32)
    
    def _invoke(self, netin):
        _, r_vec, t_vec = cv2.solvePnP(self.object_pts, 
                                       netin,
                                       self.cam_mat, 
                                       self.dist_coeffs)
        return r_vec, t_vec
    
    def _get_face_in_cam(self, trans):
        full_face = get_face3d()
        temp_one = np.ones((full_face.shape[0], 1)).astype(np.float32)
        appended_object_pts = np.concatenate((full_face, temp_one), axis=1).T
        face_3d_in_cam = np.dot(trans, appended_object_pts)
        return face_3d_in_cam[:3, :]
    
    def _post_processing(self, r_vec, t_vec, face, print_euler):
        """
        netout r_vec, t_vec
        1. convert r_vec to r_mat
        2. get landmarks_in_cam
        """
        r_mat = rodrigues(r_vec)
        # collect face info
        face.r_mat = r_mat
        face.t_vec = t_vec
        face.r_vec = r_vec
        face.world_to_cam = get_trans(r_mat, t_vec)
        face.euler_angle = rmat_to_euler(r_mat)
        if print_euler:
            print("pitch: {}, yaw: {}, roll: {}".format(face.euler_angle[0], face.euler_angle[1], face.euler_angle[2]), end='\r')
        face.landmarks_in_cam = self._get_face_in_cam(face.world_to_cam)
    
    def inference(self, image_info, print_euler=False):
        image_info.cam_mat = self.cam_mat
        image_info.dist_coeffs = self.dist_coeffs
        if len(image_info.faces) > 0:
            for face in image_info.faces:
                netin = self._preprocess(face)
                netout = self._invoke(netin)
                self._post_processing(netout[0], netout[1], face, print_euler)

