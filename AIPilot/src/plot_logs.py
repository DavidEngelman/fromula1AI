import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_loss = event_acc.Scalars("Loss/train")
    validation_loss = event_acc.Scalars("Loss/validation")

    training_loss = [elem[2] for elem in training_loss]
    validation_loss = [elem[2] for elem in validation_loss]


    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation accuracy')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = "..\logs\cnn_keyboard_1\events.out.tfevents.1595846361.DESKTOP-CND0H6K.9668.0"
    plot_tensorflow_log(log_file)