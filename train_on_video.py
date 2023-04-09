import os
import cv2
import numpy as np
import onnxruntime as ort


def process3(results):
    boxes, scores = results[:4, :].T.astype(int), results[4, :]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.4, 0.5)
    if len(indices) > 0:
        return boxes[indices]
    else:
        return []


session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])
for file in os.listdir('Videos/frames/'):
    image = cv2.imread('Videos/frames/'+file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640)).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    boxes = session.run(None, {'images': image})[0][0]
    boxes = process3(boxes)
    if len(boxes) > 0:
        # with open('Videos/frames/txt/'+file[:-4]+'.txt')
        # print('Videos/frames/txt/'+file[:-4]+'.txt')
        with open('Videos/frames/txt/'+file[:-4]+'.txt', 'w') as text:
            for index in boxes:
                result = np.around(index / [640, 640, 640, 640], decimals=6)
                text.write(f'0 {" ".join(str(i) for i in result)}\n')

