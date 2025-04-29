BAG_FILE_PATH = "bag.bag"


OUT_PATH = BAG_FILE_PATH.split("/")[-1].split(".")[0]
NAO = False

from cameras.depth_camera import RGDFrameIterator
from utils.config import *
import cv2
import numpy as np
from gaze.gazelle.utils import draw_gaze_lines, draw_gaze_arrow
from detection.yolo_world import YoloWorld

from gaze.panoramical_gaze import Gazelle
gazelle_model = Gazelle()


previous_faces = {}
face_counter = 0

def compute_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

TRAINING = False
DEBUG_SHOW = True
model = YoloWorld()
model.set_classes(["toy", "horse toy"])

rgdb_data = RGDFrameIterator(BAG_FILE_PATH)
while True:

        rgb, depth = next(rgdb_data)
        try:
            areas = []
            res= gazelle_model.estimate_external_gaze(rgb)
            current_faces = {}
            for result in res.keys():
                lines = res[result]['line']
                face_bbox = res[result]['face'] 
                area = compute_area(face_bbox) 
                areas.append(area)
            #Compute only the biggest face
            face_idx = areas.index(max(areas))
            if DEBUG_SHOW:
                print(face_idx)
            lines = res[face_idx]['line']
            face_bbox = res[face_idx]['face']
            color_image, heatmap_image = draw_gaze_arrow(rgb, res[face_idx])

            robot_bboxes, robot_classes, robot_confidences = model.predict(color_image)
            start_point = (int(lines['start'][0]), int(lines['start'][1]))
            end_point = (int(lines['end'][0]), int(lines['end'][1]))
            depth_start = rgdb_data.get_3d_point(depth, start_point[0], start_point[1])
            depth_end = rgdb_data.get_3d_point(depth, end_point[0], end_point[1])
            distance = np.linalg.norm(np.array(depth_start) - np.array(depth_end))
            x1 = face_bbox[0] * color_image.shape[1]
            x2 = face_bbox[2] * color_image.shape[1]
            y1 = face_bbox[1] * color_image.shape[0]
            y2 = face_bbox[3] * color_image.shape[0]
            #PUT DISTANCE IN THE MIDDLE OF THE LINE
            face = rgb[int(y1):int(y2), int(x1):int(x2)]
            if TRAINING:
                cv2.imwrite(f"train/face_{face_counter}.jpg", rgb)
            #save label in a YOLO format
            face_center = (int((x1 + x2) / 2), int((y1 + y2) / 2)) 
            face_width = x2 - x1
            face_height = y2 - y1
            #Se esiste un volto
            if face is not None and TRAINING:
                with open(f"train/face_{face_counter}.txt", "w") as f:
                    f.write(f"0 {face_center[0] / color_image.shape[1]} {face_center[1] / color_image.shape[0]} {face_width / color_image.shape[1]} {face_height / color_image.shape[0]}")
            mid_x = (start_point[0] + end_point[0]) // 2
            mid_y = (start_point[1] + end_point[1]) // 2
            if DEBUG_SHOW:
                cv2.putText(color_image, f"Distance: {distance:.2f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x1, y1, x2, y2 = face_bbox
            #emotion = (f"Face {detect_emotion_from_face_array(face)}")
            emotion = ""
            #Put emotion at the top right
            if DEBUG_SHOW:
                cv2.putText(color_image, emotion, (int(x1 * color_image.shape[1]), int(y1 * color_image.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(color_image, start_point, (int(lines['end'][0]), int(lines['end'][1])), (0, 0, 255), 2)
            face_counter += 1
            #Save a csv file with the distance with True flag if end point is in the box of toy detected by yolo world otherwise False
            if DEBUG_SHOW:
                print(end_point)
            gaze_robot = False

            for box in robot_bboxes:
                if DEBUG_SHOW:
                    cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                threshold = 30  # Define a threshold for the area around the box
                if (int(lines['end'][0]) > box[0] - threshold and int(lines['end'][0]) < box[2] + threshold and 
                    int(lines['end'][1]) > box[1] - threshold and int(lines['end'][1]) < box[3] + threshold):
                    gaze_robot = True
                    print("Gaze on ", robot_classes[robot_bboxes.index(box)])

                    if DEBUG_SHOW:
                        cv2.putText(color_image, f"Gaze on {robot_classes[robot_bboxes.index(box)]}", (int(lines['end'][0]), int(lines['end'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    obj_seen = robot_classes[robot_bboxes.index(box)]
                    with open("out/"+ OUT_PATH+ ".csv", "a") as f:
                        print("saving")
                        f.write(f"{distance},{obj_seen}, {emotion}\n")
        
            with open("out/"+ OUT_PATH+ ".csv", "a") as f:
                print("saving")
                f.write(f"{distance}, None , {emotion}\n")
        except Exception as e:
            print(e)
            continue
        if DEBUG_SHOW:
            cv2.imshow("Tracked Faces", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()

