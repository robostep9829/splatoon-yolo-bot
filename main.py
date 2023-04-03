import time
import win32api
import numpy as np
import win32gui
import win32ui
import win32con
import time
import torch
from ultralytics import YOLO
import multiprocessing
import math
import dxcam
import cv2
import onnxruntime as ort


w = 1280  # set this
h = 720  # set this
model = YOLO("best.pt")

hwnd = None
hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.694 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')
left, top, right, bottom = win32gui.GetWindowRect(hwnd)

camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(video_mode=True, region=(left, top, right, bottom))
session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])


def screenshot(result_raw):
    while True:
        image = canvas = cv2.resize(camera.get_latest_frame(), [640, 640])
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
        image /= 255.0
        image = image.unsqueeze(0)

        # prediction = model.predict(img, show=False, save=True, save_txt=True)
        prediction = session.run(['output0'], {'images': image.numpy()})[0][0]
        result_raw['main'] = prediction
        # print(prediction[0].boxes.xywhn)
        # readData(result_raw)


def readData(result_raw):
    while True:
        boxes = []
        for number in range(0, 8400):
            if result_raw['main'][4][number] >= 0.45:
                boxes.append([result_raw['main'][0][number]/640, result_raw['main'][1][number]/640, result_raw['main'][2][number]/640, result_raw['main'][3][number]/640])
        if is_caps_lock_on():
            # print(result_raw['main'][0].boxes.xywhn.tolist())
            if boxes != []:
                for index in boxes:
                    print(index)
                    if not (math.isclose(index[0], 0.5, abs_tol=0.08) and math.isclose(index[1], 0.81, abs_tol=0.08)):
                        keyDirection(index[0], index[1])
                    else:
                        print('released-0')
                        win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0)  # I
                        win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0)  # J
                        win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)  # K
                        win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)  # L

            time.sleep(0.05)
            win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)
        else:
            print('released-2')
            # win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0)
            # win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0)
            # win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)
            # win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)


def keyDirection(player_x, player_y):
    if -0.5 < player_x - 0.5 < -0.02:
        # print('left')
        if -0.5 < player_y - 0.5 < -0.02:
            # print('left-up')
            win32api.keybd_event(0x49, 0, 0, 0)  # I
            win32api.keybd_event(0x4A, 0, 0, 0)  # J
        elif 0.02 < player_y - 0.5 < 0.5:
            # print('left-down')
            win32api.keybd_event(0x4A, 0, 0, 0)  # J
            win32api.keybd_event(0x4B, 0, 0, 0)  # K
    elif 0.02 < player_x - 0.5 < 0.5:
        # print('right')
        if -0.5 < player_y - 0.5 < -0.02:
            # print('right-up')
            win32api.keybd_event(0x49, 0, 0, 0)  # I
            win32api.keybd_event(0x4C, 0, 0, 0)  # L
        elif 0.02 < player_y - 0.5 < 0.5:
            # print('right-down')
            win32api.keybd_event(0x4B, 0, 0, 0)  # K
            win32api.keybd_event(0x4C, 0, 0, 0)  # L


def is_caps_lock_on():
    # Get the state of the caps lock key
    state = win32api.GetKeyState(win32con.VK_CAPITAL)
    # Check if the high-order bit is 1 (caps lock is on)
    return state & 0x0001 != 0


# When everything done, release the capture
if __name__ == '__main__':
    with multiprocessing.Manager() as m:
        results_raw = m.dict()
        rec = multiprocessing.Process(target=screenshot, args=(results_raw, ))
        rec.start()
        time.sleep(3)
        read = multiprocessing.Process(target=readData, args=(results_raw, ))
        read.start()
        rec.join()
        read.join()
