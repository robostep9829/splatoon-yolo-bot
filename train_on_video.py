import os
import cv2
import numpy
import numpy as np
import onnxruntime as ort
import subprocess
import shutil

filename = 'Splatoon 3 Online Turf Wars ｜ Livestream [No Commentary] [_kSw_heA4Kk].webm'
directory = 'video3'

def process3(results):
    boxes, scores = results[:4, :].T.astype(int), results[4, :]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.4, 0.5)
    if len(indices) > 0:
        return set(map(tuple, boxes[indices]))
    else:
        return []


session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])
frames = []
for file in os.listdir('Videos/'+directory):
    image = cv2.imread('Videos/'+directory+'/'+file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640)).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    print(str(file)+' ', end='\r', flush=True)
    boxes = session.run(None, {'images': image})[0][0]
    boxes = process3(boxes)
    if len(boxes) > 1:
        frames.append(file[:-4])
        # with open('Videos/frames/txt/'+file[:-4]+'.txt')
        # print('Videos/frames/txt/'+file[:-4]+'.txt')
        with open('Videos/'+directory+'/txt/'+file[:-4]+'.txt', 'w') as text:
            for index in boxes:
                result = np.around(numpy.array(index) / [640, 640, 640, 640], decimals=6)
                text.write(f'0 {" ".join(str(i) for i in result)}\n')
        #command = ['ffmpeg', '-i', 'Videos/'+filename, '-y', '-vf', f'select=\'eq(n\,{file[:-4]}\')', '-vframes', '1', f"{f'Videos/{directory}/extracted'}/{file[:-4]}.png"]
        #subprocess.call(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        shutil.copy('Videos/'+directory+str(file),'Videos/'+directory+'/'+'extracted/'+str(file))
