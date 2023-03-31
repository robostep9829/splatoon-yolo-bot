import time
import win32api
import numpy as np
import win32gui
import win32ui
import win32con
import time
from ultralytics import YOLO
import multiprocessing
import math
import torch_directml

w = 1280  # set this
h = 720  # set this
dml = torch_directml.device()
model = YOLO("best.pt")
model.to(dml)

hwnd = None
hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.689 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')


def screenshot(result_raw):
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
        prediction = model.predict(img, show=True, half=True)
        result_raw['main'] = prediction
        # print(prediction[0].boxes.xywhn)
        readData(result_raw)


def readData(result_raw):
    while True:
        if is_caps_lock_on():
            # print(result_raw['main'][0].boxes.xywhn.tolist())
            if result_raw['main'][0].boxes.xywhn != []:
                for index in result_raw['main'][0].boxes.xywhn.tolist():
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
        break


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
        # read = multiprocessing.Process(target=readData, args=(results_raw, results_shared))
        # read.start()
        rec.join()
        # read.join()
