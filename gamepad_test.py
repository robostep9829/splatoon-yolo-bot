import vgamepad as vg
import time
from win32api import GetKeyState

gamepad = vg.VX360Gamepad()


def input1():
    for i in range(0, 100):
        gamepad.right_trigger_float(value_float=i / 100)
        gamepad.left_trigger_float(value_float=i / 100)
        gamepad.update()
        time.sleep(0.005)


while True:
    if GetKeyState(0x51) < 0:
        input1()


gamepad.reset()
gamepad.update()
