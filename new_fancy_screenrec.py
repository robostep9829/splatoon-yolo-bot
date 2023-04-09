import cv2
import dxcam
import win32gui
import onnxruntime as ort
import numpy as np
import torch


def process3(results):
    boxes, scores = results[:4, :].T.astype(int), results[4, :]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
    if len(indices) > 0:
        return boxes[indices]
    else:
        return []


hwnd = None
hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.700 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')
left, top, right, bottom = win32gui.GetWindowRect(hwnd)

camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(video_mode=True, region=(left, top, right, bottom))
session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])
while True:
    image = canvas = cv2.resize(camera.get_latest_frame(), (640, 640))
    image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    outputs = session.run(['output0'], {'images': image.numpy()})
    # cv2.imshow('window', cv2.resize(camera.get_latest_frame(), (right//4, bottom//2)))
    boxess = process3(outputs[0][0])
    # Draw the final bounding boxes on the input image
    for box in boxess:
        cv2.circle(canvas, box[:2], box[3] // 2, (255, 0, 255), 3)
    # Show the final output image
    cv2.imshow("Output", canvas)
    # cv2.imshow("Output", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
camera.stop()
