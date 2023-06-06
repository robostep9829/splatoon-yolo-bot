import time
import win32api
import numpy as np
import win32gui
import win32con
import time
import torch
import multiprocessing
import math
import dxcam
import cv2
import onnxruntime as ort

w = 1280  # set this
h = 720  # set this

hwnd = None
hwnd = win32gui.FindWindow(None, 'Ryujinx 1.1.869 - Splatoon 2 v5.5.1 (0100F8F0000A2000) (64-bit)')
left, top, right, bottom = win32gui.GetWindowRect(hwnd)

camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(video_mode=True, region=(left, top, right, bottom))
session = ort.InferenceSession('best.onnx', providers=['DmlExecutionProvider'])


def screenshot(result_raw) -> None:
    while True:
        image = cv2.resize(camera.get_latest_frame(), (640, 640))
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)

        # prediction = model.predict(img, show=False, save=True, save_txt=True)
        result_raw['main'] = session.run(['output0'], {'images': image.numpy()})[0][0]
        # result_raw['main'] = prediction
        # print(prediction[0].boxes.xywhn)
        # readData(result_raw)


def readData(result_raw) -> None:
    while True:
        # boxes = result_raw['main']
        ## boxes = set(map(tuple, process3(result_raw['main'])))
        boxes = process3(result_raw['main'])
        img = np.zeros(shape=[640, 640, 3], dtype=np.uint8)
        cv2.circle(img, (320, 320), 6, [255, 255, 255], 4)
        cv2.circle(img, (320, 455), 6, [200, 200, 200], 4)
        if len(boxes) > 0:
            for u, index in enumerate(boxes):
                color = [255, 0, 0]
                if u == 0:
                    color = [255, 255, 0]
                cv2.circle(img, (int(index[0]), int(index[1])), int(index[2]), color, 4)
        cv2.imshow('image', img)
        cv2.waitKey(1)
        if is_caps_lock_on() and len(boxes) > 0:
            #for index in boxes:
                # print(index)
                # box = np.array(index) / [640, 640, 640, 640]
                # print(box)
                #if not (math.isclose(index[0]/640, 0.5, abs_tol=0.05) and math.isclose(index[1]/640, 0.73, abs_tol=0.08)):
                    # keyDirection(index[0]/640, index[1]/640, 0.05)
                    # continue
            keyDirection(boxes[0][0]/640, boxes[0][1]/640, 0.05)
            time.sleep(0.01)


        # else:
        # print('released-2')
        # win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0) # I
        # win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0) # J
        # win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0) # K
        # win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0) # L
        # print('done')


def keyDirection(player_x: float, player_y: float, sleep: float) -> None:
    if -0.5 < player_x - 0.5 < -0.02:
        # print('left')
        if -0.5 < player_y - 0.5 < -0.02:
            # print('left-up')
            win32api.keybd_event(0x49, 0, 0, 0)  # I
            win32api.keybd_event(0x4A, 0, 0, 0)  # J
            time.sleep(sleep)
            win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0)  # I
            win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0)  # J
        elif 0.02 < player_y - 0.5 < 0.5:
            # print('left-down')
            win32api.keybd_event(0x4A, 0, 0, 0)  # J
            win32api.keybd_event(0x4B, 0, 0, 0)  # K
            time.sleep(sleep)
            win32api.keybd_event(0x4A, 0, win32con.KEYEVENTF_KEYUP, 0)  # J
            win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)  # K
    elif 0.02 < player_x - 0.5 < 0.5:
        # print('right')
        if -0.5 < player_y - 0.5 < -0.02:
            # print('right-up')
            win32api.keybd_event(0x49, 0, 0, 0)  # I
            win32api.keybd_event(0x4C, 0, 0, 0)  # L
            time.sleep(sleep)
            win32api.keybd_event(0x49, 0, win32con.KEYEVENTF_KEYUP, 0)  # I
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)  # L
        elif 0.02 < player_y - 0.5 < 0.5:
            # print('right-down')
            win32api.keybd_event(0x4B, 0, 0, 0)  # K
            win32api.keybd_event(0x4C, 0, 0, 0)  # L
            time.sleep(sleep)
            win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)  # K
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)  # L


def is_caps_lock_on() -> bool:
    # Get the state of the caps lock key
    state = win32api.GetKeyState(win32con.VK_CAPITAL)
    # Check if the high-order bit is 1 (caps lock is on)
    return state & 0x0001 != 0


def process3(results: np.ndarray) -> np.ndarray:
    boxes, scores = results[:4, :].T.astype(int), results[4, :]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
    if len(indices) > 0:
        retval = boxes[indices]
        index_to_remove = np.argmin(np.linalg.norm(retval[:, :2] - [320, 455], axis=1))
        retval = np.delete(retval, index_to_remove, axis=0)
        distances = np.linalg.norm(retval[:, :2] - [320, 320], axis=1)
        return retval[np.argsort(distances)]
    else:
        return []


# When everything done, release the capture
if __name__ == '__main__':
    with multiprocessing.Manager() as m:
        results_raw = m.dict()
        rec = multiprocessing.Process(target=screenshot, args=(results_raw,))
        rec.start()
        time.sleep(3)
        read = multiprocessing.Process(target=readData, args=(results_raw,))
        read.start()
        rec.join()
        read.join()
