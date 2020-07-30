import keyboard
import d3dshot
import argparse
import time

from PIL import Image
from torchvision import transforms

import torch

from model import CNN




def preprocess_data(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame = transform(frame)
    return frame

def translate_input(input):
    input = input.tolist()[0]
    keys = []
    threshold = 0.25
    if input[0] > threshold:
        keys.append("z")
    if input[1] > threshold:
        keys.append("q")
    if input[2] > threshold:
        keys.append("s")
    if input[3] > threshold:
        keys.append("d")

    return "+".join(keys)


class AIplayer:
    def __init__(self, model):
        self.model = model
        self.screenshoter = d3dshot.create()
        init_img = Image.open("../f1_data/test/0000_0aec6b1c-ab42-4af5-8e25-c913152042fd.png")
        self.model(torch.unsqueeze(preprocess_data(init_img), 0).cuda())
    def getInput(self, frame):
        return self.model(torch.unsqueeze(preprocess_data(frame), 0).cuda())

    def release_all(self):
        keyboard.release("z+q+s+d")

    def perform_action(self, input):
        keys_to_press = translate_input(input)
        self.release_all()
        if keys_to_press != "":
            print(f"Pressing {keys_to_press}")
            keyboard.press(keys_to_press)



    def get_img(self):
        img = self.screenshoter.screenshot().resize((480, 270), Image.ANTIALIAS)
        return img

    def play(self):
        start_key = "a"
        stop_key = "r"
        print("Press 'a' to start the AI player")
        c = 0
        while True:
            if keyboard.is_pressed(start_key):
                print("playing... press 'r' to stop")
                while True:
                    if True:
                        print(c)
                        img = self.get_img()
                        print("frame received")

                        start = time.time()
                        _input = self.getInput(img)
                        print(f"time to perform: {time.time()- start}")

                        print(f"model input: {_input}")
                        self.perform_action(_input)
                    c += 1

                    if keyboard.is_pressed(stop_key):
                        self.release_all()
                        print("Stop, Press 'a' to restart ")
                        break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    print("Loading model...")
    model = CNN()
    with open(args.model, 'rb') as f:
        checkpoint = torch.load(f)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    print("Model ready.")




    player = AIplayer(model)
    player.play()