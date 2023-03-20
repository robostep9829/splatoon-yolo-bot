from ultralytics import YOLO
import torch
import torch_directml

dml = torch_directml.device()
model = YOLO("yolov8s.yaml")
model.train(data="train.yaml", epochs=5)