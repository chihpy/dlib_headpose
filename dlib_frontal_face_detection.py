import dlib
import cv2

class Face_detection:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
    def post_process(self, rects):
        boxes = []
        for rect in rects:
            xmin = rect.left()
            xmax = rect.right()
            ymin = rect.top()
            ymax = rect.bottom()
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes
        
    def inference(self, image):
        face_rects = self.detector(image, 0)
        boxes = self.post_process(face_rects)
        return face_rects, boxes
    
    def draw_face(self, image, boxes):
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)