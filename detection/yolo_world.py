from ultralytics import YOLOWorld
import cv2
from utils.config import *  # Importa config dal pacchetto creta.utils

class YoloWorld:
      
    def __init__(self):
          self.model = YOLOWorld(YOLO_PATH)
          self.classes = None

    def set_classes(self, classes):
        self.model.set_classes(classes)

    def predict(self, image):
        results = self.model.predict(image, verbose=False, conf=0.1)
        bboxes = []
        classes = []
        confidences = []
        for result in results:
            detections = result.boxes.cpu().numpy()  
            for detection in detections:                
                bbox = detection.xyxy[0]
                id_class = result.names[detection.cls[0]]
                confidence = detection.conf[0]
                bboxes.append(bbox)
                classes.append(id_class)
                confidences.append(confidence)
        return bboxes, classes, confidences

    def get_image_with_bboxes(self, image, conf=0.1):
        bboxes, classes, confidences = self.predict(image)
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            class_id = classes[i]
            confidence = confidences[i]
            if confidence > conf:
                cv2.putText(image, class_id + " "+ str(confidence), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,  1, (250,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (250,0,0), 2)
        return image

