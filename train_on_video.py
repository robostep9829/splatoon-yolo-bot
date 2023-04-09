#from ultralytics import YOLO
import os

import cv2
import numpy as np
import onnxruntime as ort
#model = YOLO("datasets/FromRoboflow/best.pt")
#model.predict('runs/detect/predict2/Frames 2', hide_labels=True, box=False, show=False, save=False, save_txt=True, line_thickness=1)
session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])
for file in os.listdir('Videos/frames/'):
    image = cv2.imread('Videos/frames/'+file)
    #cv2.imshow('aboba', image)
    #cv2.waitKey(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = np.array(image).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    boxes = session.run(None, {'images': image})[0][0]
    boxes = boxes[:, np.where(boxes[4, :] > 0.49)[0]]
    if boxes.shape[-1] > 0:
        # with open('Videos/frames/txt/'+file[:-4]+'.txt')
        # print('Videos/frames/txt/'+file[:-4]+'.txt')
        with open('Videos/frames/txt/'+file[:-4]+'.txt', 'w') as text:
            for index in range(boxes.shape[-1]):
                result = boxes[:4, index] / [640, 640, 640, 640]
                result = np.around(result, decimals=6)
                text.write('0 ')
                for i in result:
                    text.write(str(i)+' ')
                text.write('\n')

