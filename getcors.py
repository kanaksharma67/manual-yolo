import pyautogui
import time

print("Move mouse to element corners and press Ctrl+C to stop:")
try:
    while True:
        x, y = pyautogui.position()
        print(f"X: {x}, Y: {y}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Done")

