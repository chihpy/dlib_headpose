import dlib
import cv2


class Landmark_detection_68:
    def __init__(self, model_dat):
        self.landmark_model = dlib.shape_predictor(model_dat)
    
    def post_process(self, shape):
        pts = []
        for pt in shape.parts():
            pts.append((pt.x, pt.y))
        return pts
        
    
    def inference(self, image, face_rect):
        shape = self.landmark_model(image, face_rect)
        pts = self.post_process(shape)
        return pts
    
    def draw_pts(self, image, pts):
        for (x, y) in pts:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)