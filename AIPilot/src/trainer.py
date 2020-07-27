import argparse
import torch
import time
import logging
import sys
import os

from torchvision import datasets, models, transforms

import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import F1Dataset
from model import CNN


class Trainer(object):
    def __init__(self, model, output_name, train_loader=None, val_loader=None):
        self.model = model
        # self.criterion = nn.NLLLoss()
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters())

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print("using cuda")
            self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.log = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        self.writer = SummaryWriter(f"../logs/{output_name}")

        self.print_size = 10
        self.output = output_name

    def loop(self, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + epochs):
            print("Starting Epoch: ", epoch)
            self.train(epoch)
            self.val(epoch)
            self.checkpoint(epoch)

    def checkpoint(self, epoch):
        if not os.path.exists('../checkpoints'):
            os.makedirs('../checkpoints')

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }

        torch.save(state, "../checkpoints/{}_{}.pt".format(self.output, epoch))

    def train(self, epoch):

        self.model.train()
        start_time = time.time()
        epoch_loss = 0.0
        running_loss = 0.0
        for it, sample_batched in enumerate(self.train_loader):
            it_start = time.time()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            input_batch, labels = sample_batched
            if torch.cuda.is_available():
                input_batch = input_batch.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = self.model(input_batch)
            # print(outputs.shape)
            # print(labels.long().squeeze_.shape)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()


            running_loss += loss.item()
            epoch_loss += loss.item()
            total_time = time.time() - it_start

            if it % self.print_size == 0:  # print every
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'iteration': it + 1, 'n_iteration': len(self.train_loader),
                    'time': round(total_time, 3),
                    'loss': running_loss / self.print_size,
                })
                running_loss = 0.0

            self.writer.add_scalar("Loss/iteration", loss.item(), ((epoch) * len(self.train_loader)) + it)

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch,
            'loss': epoch_loss / len(self.train_loader),
            'time': round(time.time() - start_time, 1),
        })
        self.writer.add_scalar("Loss/train", epoch_loss / len(self.train_loader), epoch)

    def val(self, epoch):
        self.model.eval()
        running_loss = 0.0
        start_time = time.time()

        for it, sample_batched in enumerate(self.val_loader):
            # get the inputs; data is a list of [inputs, labels]
            input_batch, labels = sample_batched
            if torch.cuda.is_available():
                input_batch = input_batch.cuda()
                labels = labels.cuda()
            # forward
            outputs = self.model(input_batch)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': running_loss / len(self.val_loader),
            'time': round(start_time - time.time(), 1),
        })
        self.writer.add_scalar("Loss/validation", running_loss / len(self.val_loader), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="path of the logging file", required=True)
    parser.add_argument("--data_root", help="path of the data file", required=True)
    parser.add_argument("--checkpoint", help="path of the checkpoint file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_classes", type=int, default=4)

    args = parser.parse_args()


    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = F1Dataset("{}/train/".format(args.data_root), transforms=transformations,
                              n_classes=args.n_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = F1Dataset("{}/val/".format(args.data_root), transforms=transformations,
                            n_classes=args.n_classes)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if not os.path.exists('../trained_models'):
        os.makedirs('../trained_models')

    output = '../trained_models/{}'.format(args.output)

    pretrained = models.resnet50(pretrained=True)

    cnn = CNN(pretrained, n_classes=4)
    print(type(cnn))
    start_epoch = 0

    if args.checkpoint:
        with open(args.checkpoint, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        start_epoch = checkpoint['epoch']
        cnn.load_state_dict(pretrained_state_dict)



    trainer = Trainer(
        model=cnn,
        output_name=args.output,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
    )

    trainer.loop(epochs=20, start_epoch=start_epoch)
    torch.save(trainer.model.state_dict(), "{}.pt".format(output))


if __name__ == '__main__':
    main()
