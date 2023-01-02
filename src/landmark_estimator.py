"""
TODO: model_path problem
"""
import os
import cv2
import dlib

class LandmarkEstimator:
    def __init__(self, name):
        self.name = name
        if name == 'dlib_v1':
            self.model_path = r"/home/pymi/dlib_project/dlib_headpose/models/shape_predictor_68_face_landmarks.dat"
            print("using dlib_landmark68_v1 model, user should check the face detector must be dlib front model")
            self.model = dlib_landmark68(name, self.model_path)
        elif name == 'dlib_v2':
            self.model_path = r"/home/pymi/dlib_project/dlib_headpose/models/shape_predictor_68_face_landmarks_GTX.dat"
            print("using dlib_landmark68_v2, user should check the face bounding box must be square")
            self.model = dlib_landmark68(name, self.model_path)
    def inference(self, image, image_info, from_cv2=True):
        self.model.inference(image, image_info, from_cv2)


class dlib_landmark68():
    def __init__(self, name, model_path):
        if not os.path.exists(model_path):
            raise("model_path unexists: "+ model_path)
        self.estimator = dlib.shape_predictor(model_path)
        self.name = name

    def _preprocess(self, image, from_cv2=True):
        if from_cv2:
            netin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            netin = image.copy()
        return netin

    def _invoke(self, netin, rect):
        netout = self.estimator(netin, rect)
        return netout
    
    def _post_process(self, netout, face):
        pts = []
        face.landmark_model_name = self.name
        for pt in netout.parts():
            pts.append((pt.x, pt.y))
        face.landmarks = pts
    
    def inference(self, image, image_info, from_cv2=True):
        netin = self._preprocess(image, from_cv2)
        for rect, face in zip(image_info.dlib_rects, image_info.faces):
            netout = self._invoke(netin, rect)
            self._post_process(netout, face)




    

