from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("datasets/FromRoboflow/best.pt")
model.predict('https://www.youtube.com/watch?v=lBq7tNui-9g', hide_labels=True, box=False, show=True, save=False, save_txt=False, line_thickness=1)