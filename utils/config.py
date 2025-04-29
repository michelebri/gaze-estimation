import os


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

WEIGHTS_PATH = os.path.join(DIR_PATH, "weights")
YOLO_PATH = os.path.join(WEIGHTS_PATH, "yolov8x-worldv2.pt")
GAZELLE_PATH = os.path.join(WEIGHTS_PATH, "gazelle_dinov2_vitl14_inout.pt")
YOLO_FACE_PATH = "/home/michele/Desktop/UNINT_NAO/Topublish/gaze-estimation/utils/weights/best.pt"

