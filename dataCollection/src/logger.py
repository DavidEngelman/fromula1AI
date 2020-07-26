import time
import keyboard
import d3dshot
import uuid
import threading
import os
# import cv2
import pygame
import pandas as pd
from PIL import Image
import datetime
import mss

from src.conf import IMG_FOLDER, KEYSTROKES_FILENAME, TRACKED_KEYS


class AsyncWrite(threading.Thread):

    def __init__(self, data, out):
        # calling superclass init
        threading.Thread.__init__(self)
        self.data = data
        self.out = out
        self.logs = {"id": [], "keystrokes": []}

    def run(self):
        ts = str(datetime.datetime.now().timestamp()).replace('.', '')
        output_path = os.path.join(self.out, ts)
        os.mkdir(output_path)

        for img, inputs in self.data:
            # res = img.resize((480, 270), Image.ANTIALIAS)
            inputs_str = "_".join(inputs)
            img.save(output_path + f"/{inputs_str}.png")

        print("Finished background save")


class MainLogger:
    def __init__(self, freq, ouput_dir_path):
        self.output_path = ouput_dir_path
        self.freq = freq
        self.tracked_keys = TRACKED_KEYS
        self.screenshoter = d3dshot.create()

    def get_img(self):
        return self.screenshoter.screenshot().resize((480, 270), Image.ANTIALIAS)

    def get_keystrokes(self):
        pass

    def log(self):
        start_key = "a"
        stop_key = "r"
        print("Press 'a' to start logging")
        background = None
        while True:
            if keyboard.is_pressed(start_key):
                print("Logging start, press 'r' to stop")
                data_buffer = []
                while True:

                    img = self.get_img()
                    keystrokes = self.get_keystrokes()
                    frame_id = str(uuid.uuid4())
                    data_buffer.append((img, keystrokes, frame_id))

                    if len(data_buffer) == 100:
                        print("saving data to disk...")
                        background = AsyncWrite(data_buffer, self.output_path)
                        background.start()
                        data_buffer = []
                    if keyboard.is_pressed(stop_key):
                        print("Logging stop, Press 'a' to restart ")
                        break

                if background:
                    background.join()


class KeyboardLogger(MainLogger):
    def __init__(self, freq, ouput_dir_path):
        MainLogger.__init__(self, freq, ouput_dir_path)

    def get_keystrokes(self):
        keystrokes = ""
        for key in self.tracked_keys:
            keystrokes += str(int(keyboard.is_pressed(key)))
        # print(keystrokes)
        return keystrokes


class G29Logger(MainLogger):
    def __init__(self, freq, ouput_dir_path):
        MainLogger.__init__(self, freq, ouput_dir_path)
        pygame.init()
        pygame.joystick.init()

        self.joystick_count = pygame.joystick.get_count()

        self.joystick = pygame.joystick.Joystick(0)

        self.joystick.init()
        joy_name = self.joystick.get_name()
        print(joy_name)  # <----detected and correct name
        print(self.joystick.get_numaxes())
        print(self.joystick.get_axis(1))

    def log(self):

        recording = False
        buffer = []
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        c = 0
        while True:
            for event in pygame.event.get():
                # If a button is pressed
                # if event.type == pygame.JOYBUTTONDOWN:
                #     print(f"Joystick button pressed {c} times")
                #     c+=1

                # If x is pressed and we are not yet recording
                if joystick.get_button(0) and not recording:
                    print("Start recording")
                    recording = True
                    buffer = []
                # If x is pressed and we are recording
                # elif joystick.get_button(0) and recording:
                #     print("Start new recording")
                #     background = AsyncWrite(buffer, self.output_path)
                #     background.start()
                #     buffer = []
                #     recording = True
                # If # is pressed and we are recording
                elif joystick.get_button(1) and recording:
                    print("Stop recording")
                    background = AsyncWrite(buffer, self.output_path)
                    background.start()
                    buffer = []
                    recording = False

                if recording and c % 20 == 0:

                    axes = joystick.get_numaxes()
                    image = self.get_img()
                    inputs = [str(joystick.get_axis(i)) for i in range(axes)][:-1]
                    buffer.append((image, inputs))

                c+=1


    # def get_keystrokes(self):
    #     keystrokes = ""
    #     for key in self.tracked_keys:
    #         keystrokes += str(int(keyboard.is_pressed(key)))
    #     # print(keystrokes)
    #     return keystrokes




if __name__ == '__main__':
     g29 = G29Logger(30, "logs")
     g29.log()
     # start = time.time()
     # with mss.mss() as sct:
     #     sct_img = sct.grab(sct.monitors[1])
     # print(time.time() - start)
     #
     # screenshoter = d3dshot.create()
     # start = time.time()
     # screenshoter.screenshot()
     # print(time.time() - start)
