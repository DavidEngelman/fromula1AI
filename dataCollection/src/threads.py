import threading

from PIL import Image
import d3dshot
import pandas
import sys
import pygame
import time
import keyboard
import os
import copy
from screen_recorder_sdk import screen_recorder


import socket
from f1_2020_telemetry.packets import unpack_udp_packet, PacketID

import datetime

"""
3 threads: 
    ScreenRecord: record the screen
    UPDListener: intercept udp packets for f12020 telemetry
    G29Logger: Log the g29 controller inputs
"""
STUFF="hi"
precision = 12

def is_time(start, c):
    curr_str = str(datetime.datetime.now().timestamp())[:12]
    wait_str = str(start + (1/5) * c)[:12]
    # print("current:", datetime.datetime.fromtimestamp(float(curr_str)))
    # print("waiting for: ", datetime.datetime.fromtimestamp(float(wait_str)))
    # print(str(datetime.datetime.now().timestamp())[:12] == str(start + (1/5) * c)[:12])
    # print(curr_str == wait_str)
    return curr_str == wait_str



class ScreenRecord(threading.Thread):

    def __init__(self, out, start):
        threading.Thread.__init__(self)
        self.img_buffer = []
        self.out = out
        self.start_time = start
        self.screenshoter = d3dshot.create()

    def get_img(self):
        return self.screenshoter.screenshot()

    def run(self):
        c = 0
        print("Recording thread running...")
        global stop
        while True:
            s = time.time()
            if is_time(start_time, c):
                name = str(datetime.datetime.now().timestamp())[:12].replace('.', '_') + '.png'
                c += 1
                img = self.get_img()
                data = {
                    "img": img
                }
                data["clock"] = str(datetime.datetime.now().timestamp())[:12]
                data["img_name"] = name
                self.img_buffer.append(data)

                print("adding img")

            if stop:
                break


        self.save()


    def save(self):
        print("saving screen data...")
        # df_data = copy.deepcopy(data)
        # del df_data["img"]
        for data in self.img_buffer:

            data['img'].resize((480, 270), Image.ANTIALIAS).save(f"../logs/{self.out}/img/{data['img_name']}")

        print("done saving screen data...")


class UPDListener(threading.Thread):

    def __init__(self, out, start):
        threading.Thread.__init__(self)
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_socket.bind(('', 20777))
        self.telemetry_buffer = []
        self.out = out
        self.player_car = 0
        self.dir_name = "udp"
        self.start_time = start



    def run(self):
        print("Listening for udp packets...")
        c=0
        got_data = False
        while True:
            udp_packet = self.udp_socket.recv(2048)
            packet = unpack_udp_packet(udp_packet)

            current_frame_data = dict()
            current_frame_data[PacketID(packet.header.packetId)] = packet

            try:
                speed = current_frame_data[PacketID.CAR_TELEMETRY].carTelemetryData[self.player_car].speed
                data = {
                    "clock": str(datetime.datetime.now().timestamp()).replace('.', '_'),
                    "telemetry": speed,
                }
                got_data = True
            except:
                pass

            if is_time(start_time,c) and got_data:
                data["clock"] = str(datetime.datetime.now().timestamp()).replace('.', '_')[:12]
                self.telemetry_buffer.append(data)
                # print("adding tele")
                c+=1
                got_data = False



            if stop:
                break

        self.save()

    def save(self):
        print("saving udp data...")
        df = pandas.DataFrame({
            "clock": [],
            "telemetry": [],
        })
        for data in self.telemetry_buffer:
            df = df.append({'clock': data["clock"],
                            'telemetry': data["telemetry"]
                            }, ignore_index=True)

        df.to_csv(f"../logs/{self.out}/telemetry_data.csv", index=False)
        print("done saving udp data...")
class G29Logger(threading.Thread):

    def __init__(self, out, start):
        # calling superclass init
        threading.Thread.__init__(self)
        self.input_buffer = []
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.axes = self.joystick.get_numaxes()
        self.start_time = start
        self.out = out
        self.dir_name = "g29"

    def run(self):
        c = 0
        print("Logging g29 data ...")
        got_data = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    inputs = [str(self.joystick.get_axis(i)) for i in range(self.axes)][:-1]
                    data = {
                        "clock": str(datetime.datetime.now().timestamp()).replace('.', '_'),
                        "inputs": inputs
                    }
                    got_data = True

            if is_time(start_time, c) and got_data:
                data["clock"] = str(datetime.datetime.now().timestamp()).replace('.', '_')[:12]
                self.input_buffer.append(data)
                # print("adding g29")
                c += 1
                got_data = False

            if stop:
                break

        self.save()

    def save(self):
        print("saving g29 data...")
        df = pandas.DataFrame({
            "clock": [],
            "inputs": [],
        })
        for data in self.input_buffer:
            df = df.append({'clock': data["clock"],
                            'inputs': data["inputs"]
                            }, ignore_index=True)

        df.to_csv(f"../logs/{self.out}/g29_data.csv", index=False)
        print("done saving g29 data...")


if __name__ == '__main__':


    stop = True

    start_key= "a"
    stop_key= "r"




    print("Press 'a' to start logging")
    background = None
    while True:
        if keyboard.is_pressed(start_key):

            log_dir_name = str(datetime.datetime.now().timestamp()).replace('.', '')
            os.mkdir(f"../logs/{log_dir_name}")
            os.mkdir(f"../logs/{log_dir_name}/img")

            stop = False

            start_time = datetime.datetime.now().timestamp() + 5
            screenThread = ScreenRecord(log_dir_name, start_time)
            updThread = UPDListener(log_dir_name, start_time)
            g29Thread = G29Logger(log_dir_name, start_time)

            screenThread.start()
            updThread.start()
            g29Thread.start()

            while True:
                if keyboard.is_pressed(stop_key):
                    stop = True
                    print("Logging stop, Press 'a' to restart ")

                    break

