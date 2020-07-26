import os
import pandas as pd
from shutil import move
from sklearn.model_selection import train_test_split


def copy(p1, p2):
    file = open(p2, "wb")
    with open(p1, "rb") as f:
        while True:
            byte = f.read(1)
            if not byte:
                break
            file.write(byte)


def create_dataset(data_source, d_name="dataset"):
    df_inputs = pd.read_csv(f"{data_source}/keystrokes.csv", header=None, dtype=str)
    df_inputs = df_inputs.drop_duplicates(subset=1, keep='first', inplace=False)
    os.makedirs(name)
    i = 0
    for index, row in df_inputs.iterrows():
        if i % 1000 == 0: print(i)
        uuid, inputs = row[1], row[2]
        try:
            copy(f"{data_source}/img/{uuid}.png", f"{name}/{inputs}_{uuid}.png")
        except:
            pass
        i += 1


def split_data(d_name, val=0.2, test=0.2):
    files = [f for f in os.listdir(d_name)]
    x_train, x_test = train_test_split(files, test_size=test)
    x_train, x_val = train_test_split(x_train, test_size=val)
    print(len(x_train), len(x_val), len(x_test))

    os.makedirs(f"{d_name}/train")
    os.makedirs(f"{d_name}/val")
    os.makedirs(f"{d_name}/test")

    i = 0
    for dir_name, dir in zip(["train", "val", "test"], [x_train, x_val, x_test]):
        for f in dir:
            if i % 1000 == 0:
                print(i)
            move(f"{d_name}/{f}", f"{d_name}/{dir_name}/{f}")
            i += 1


if __name__ == '__main__':
    # create_dataset("../ScreenAndKeyLogger/logs/a5b41395", "f1_data")
    split_data("f1_data")
