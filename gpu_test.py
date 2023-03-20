import time
from win32api import GetKeyState
import numpy as np
import win32gui
import win32ui
import win32con
import time
from ultralytics import YOLO
import multiprocessing
import tensorflow as tf
import math
import torch_directml

w = 1280  # set this
h = 720  # set this
config = tf.ConfigProto()
config.graph_options.rewrite_options.experimental_direct_ml_ops = True
# model = YOLO("datasets/FromRoboflow/best.pt")
with tf.Session(config=config, target="directml") as sess:
    # load the YOLOv5 model
    model = tf.keras.models.load_model("path/to/yolov5/model.h5")

hwnd = None
# hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.670 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')


def screenshot():
    t0 = time.time()
    while True:
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype=np.uint8)
        img.shape = (h, w, 4)
        # cv2.imshow('frame', img)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        ex_time = time.time() - t0
        print("FPS: " + str(1 / ex_time))
        t0 = time.time()

        # prediction = model.predict(img, show=False, save=True, save_txt=True)
        prediction = model.predict(img, show=True, device='dml:0')
        # result_raw['main'] = prediction
        # print(prediction[0].boxes.xywhn)


if __name__ == '__main__':
    rec = multiprocessing.Process(target=screenshot)
    rec.start()
    rec.join()
