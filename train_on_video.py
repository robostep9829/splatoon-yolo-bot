from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("datasets/FromRoboflow/best.pt")
model.predict('runs/detect/predict2/Frames 2', hide_labels=True, box=False, show=False, save=False, save_txt=True, line_thickness=1)