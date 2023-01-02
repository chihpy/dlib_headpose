"""
TODO:
- [ ] add logging
- [ ] add decorator time test
"""
import os
import cv2
import dlib
from src.face_info import ImageInfo, FaceInfo

class FaceDetector:
    def __init__(self, name):
        self.name = name
        if name == 'dlib_frontal':
            self.model = dlib_frontal()
        elif name == "dlib_cnn":
            self.model = dlib_cnn()
        else:
            raise("unknown model name" + name)
    
    def inference(self, image, from_cv2=True):
        image_info = self.model.inference(image, from_cv2)
        return image_info

class dlib_frontal:
    def __init__(self):
        self.name = 'dlib_frontal'
        self.detector = dlib.get_frontal_face_detector()
    
    def _preprocess(self, image, from_cv2=True):
        if from_cv2:
            netin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            netin = image.copy()
        return netin
    
    def _invoke(self, netin):
        netout = self.detector(netin, 0)  # TODO: check param
        return netout
    
    def _post_process(self, netout, image_info):
        """
        convert dlib rects format to self defined bndbox format
        Args:
            netout: dlib rectangles
            image_info: class ImageInfo
        """
        image_info.dlib_rects = netout
        for rect in netout:
            xmin = int(rect.left())
            ymin = int(rect.top())
            xmax = int(rect.right())
            ymax = int(rect.bottom())
            face = FaceInfo([xmin, ymin, xmax, ymax])
            image_info.faces.append(face)

    def inference(self, image, from_cv2):
        height, width, _ = image.shape
        image_info = ImageInfo((height, width))
        netin = self._preprocess(image, from_cv2)
        netout = self._invoke(netin)
        self._post_process(netout, image_info)
        return image_info

class dlib_cnn:
    def __init__(self):
        self.name = 'dlib_cnn'
        self.model_path = "/home/pymi/dlib_project/dlib_headpose/models/mmod_human_face_detector.dat"
        assert os.path.exists(self.model_path), "model no found: "+ self.model_path
        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
    
    def _preprocess(self, image, from_cv2=True):
        if from_cv2:
            netin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            netin = image.copy()
        return netin
    
    def _invoke(self, netin):
        netout = self.detector(netin, 0)  # TODO: check param
        return netout
    
    def _post_process(self, netout, image_info):
        """
        convert dlib rects format to self defined bndbox format
        Args:
            netout: dlib rectangles
            image_info: class ImageInfo
        """
        rects = []
        for result in netout:
            rect = result.rect
            conf = result.confidence
            xmin = int(rect.left())
            ymin = int(rect.top())
            xmax = int(rect.right())
            ymax = int(rect.bottom())
            face = FaceInfo([xmin, ymin, xmax, ymax])
            face.confidence = conf
            image_info.faces.append(face)
            rects.append(rect)
        image_info.dlib_rects = rects

    def inference(self, image, from_cv2):
        height, width, _ = image.shape
        image_info = ImageInfo((height, width))
        netin = self._preprocess(image, from_cv2)
        netout = self._invoke(netin)
        self._post_process(netout, image_info)
        return image_info