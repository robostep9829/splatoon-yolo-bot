import vgamepad as vg
import time
from win32api import GetKeyState
import numpy as np
import win32gui
import win32ui
import win32con
import time
from ultralytics import YOLO
import multiprocessing
import math

w = 1280  # set this
h = 720  # set this
model = YOLO("datasets/FromRoboflow/best.pt")

hwnd = None
# hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.670 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')


def screenshot(result_raw, result_shared):
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
        prediction = model.predict(img, show=True)
        result_raw['main'] = prediction
        # print(prediction[0].boxes.xywhn)


def readData(result_raw, result_shared):
    gamepad = vg.VX360Gamepad()
    time.sleep(4)
    while True:
        #print(result_raw['main'][0].boxes.xywhn.tolist()[1])
        if result_raw['main'][0].boxes.xywhn != []:
            for index in result_raw['main'][0].boxes.xywhn.tolist():
                if not (math.isclose(index[0], 0.5, abs_tol=0.08) and math.isclose(index[1], 0.8, abs_tol=0.08)):
                    gamepad.right_joystick_float(*stickDirection(index[0], index[1]))
                else:
                    gamepad.right_joystick_float(0, 0)
        else:
            gamepad.right_joystick_float(0, 0)
        gamepad.update()


def stickDirection(player_x, player_y):
    stickx = 0
    sticky = 0
    if -0.5 < player_x - 0.5 < -0.02:
        stickx = -1
    elif 0.02 < player_x - 0.5 < 0.5:
        stickx = 1
    if -0.5 < player_y - 0.5 < -0.02:
        sticky = 1
    elif 0.02 < player_y - 0.5 < 0.5:
        sticky = -1
    return stickx, sticky


# When everything done, release the capture
if __name__ == '__main__':
    with multiprocessing.Manager() as m:
        results_raw = m.dict()
        results_shared = m.dict()
        rec = multiprocessing.Process(target=screenshot, args=(results_raw, results_shared))
        rec.start()
        time.sleep(1)
        read = multiprocessing.Process(target=readData, args=(results_raw, results_shared))
        read.start()
        rec.join()
        read.join()
