import cv2
import dxcam
import win32gui
import onnxruntime as ort
import numpy as np
import torch

input_size = [640, 640]
classes = ["player"]
colors = [[0, 255, 0]]


def postprocess(outputs, confidence_threshold=0.5, iou_threshold=0.5):
    # Extract the bounding boxes, class IDs, and confidence scores from the YOLO model output
    boxes = []
    class_ids = [0]
    scores = []
    for output in outputs:
        for detection in output:
            scores.append(detection[4])
            # class_ids.append(np.argmax(detection[0:]))
            class_ids.append(0)
            box = detection[:4]
            x, y, w, h = box.astype("int")
            # print(x, y, w, h)
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            boxes.append([x1, y1, x2, y2])
    # Apply non-maximum suppression to the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    # Create a list of the final bounding boxes, class IDs, and confidence scores
    final_boxes = []
    final_class_ids = []
    final_scores = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_class_ids.append(class_ids[i])
            final_scores.append(scores[i])
    return final_boxes, final_class_ids, final_scores


def process(result):
    boxes = []
    scores = []
    for number in range(len(result[0])):
        scores.append(result[4][number])
        x, y, w, h = result[0][number], result[1][number], result[2][number], result[3][number]
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        if False:
            print(x1, y1, x2, y2)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
    final_boxes = []
    final_scores = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
    return final_boxes, final_scores


hwnd = None
hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.694 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')
left, top, right, bottom = win32gui.GetWindowRect(hwnd)

camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(video_mode=True, region=(left, top, right, bottom))
session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])
while True:
    image = canvas = cv2.resize(camera.get_latest_frame(), [640, 640])
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
    image /= 255.0
    image = image.unsqueeze(0)
    outputs = session.run(['output0'], {'images': image.numpy()})
    # cv2.imshow('window', cv2.resize(camera.get_latest_frame(), (right//4, bottom//2)))
    boxes, scores = process(outputs[0][0])
    # Draw the final bounding boxes on the input image
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), [255, 0, 0], 2)
        # label = "{}: {:.2f}".format('player', score)
        # cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Show the final output image
    cv2.imshow("Output", canvas)
    # cv2.imshow("Output", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
camera.stop()
