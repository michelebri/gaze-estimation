import numpy as np
import cv2
import torch
from ultralytics import YOLO
from gaze.gazelle.utils import draw_gaze_lines
from utils.config import GAZELLE_PATH, YOLO_FACE_PATH
from gaze.gazelle.model import get_gazelle_model

import warnings
warnings.filterwarnings("ignore")


class Gazelle:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
        self.model.load_gazelle_state_dict(torch.load(GAZELLE_PATH, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        self.detect_faces = YOLO(YOLO_FACE_PATH)

    def obtain_faces(self, rgb_image):
        results = self.detect_faces(rgb_image)
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
        height, width, _ = rgb_image.shape
        norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]
        return norm_bboxes

    def elaborate_results(self, rgb_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
        results = {}
        for i in range(len(bboxes)):
            if i not in results:
                results[i] = {}
            bbox = bboxes[i]
            results[i]['face'] = bbox
             
            if inout_scores is not None and inout_scores[i] > inout_thresh:
                results[i]['heatmap'] = heatmaps[i].detach().cpu().numpy()  
                heatmap_np = results[i]['heatmap']
                height, width, _ = rgb_image.shape
                xmin, ymin, xmax, ymax = bbox
                max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape) 
                gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
                gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
                bbox_center_x = ((xmin + xmax) / 2) * width
                bbox_center_y = ((ymin + ymax) / 2) * height
                results[i]['line'] = {'start': (bbox_center_x, bbox_center_y), 'end': (gaze_target_x, gaze_target_y)}
        return results 

    def estimate_external_gaze(self, color_image):
        norm_bboxes = self.obtain_faces(color_image)
        img_tensor = self.transform(color_image).unsqueeze(0).to(self.device)
        if len(norm_bboxes[0]) == 0:
            return color_image
        input = {
            "images": img_tensor, # [num_images, 3, 448, 448]
            "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
        }

        with torch.no_grad():
            output = self.model(input)
        results = self.elaborate_results(color_image, output['heatmap'][0], norm_bboxes[0], output['inout'][0] if output['inout'] is not None else None, inout_thresh=0.5)
        return results
