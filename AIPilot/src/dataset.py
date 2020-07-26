import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import imageio
from PIL import Image

class F1Dataset(Dataset):
    def __init__(self, root_dir, transforms=None, n_classes=4):
        self.root = root_dir
        self.frames = os.listdir(self.root)
        self.n_classes = n_classes
        self.transforms=transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.frames[idx]
        cockpit_height = 173
        image = Image.open(self.root + filename)
        image = image.crop((0, 0, image.size[0], cockpit_height))
        if self.transforms:
            image = self.transforms(image)

        labels = [float(lb) for lb in filename.split('_')[0]]

        sample = [image, torch.tensor(labels, dtype=torch.float32)]

        return sample


if __name__ == '__main__':
    root = "f1_data/train/"

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = F1Dataset(root, transforms=transformations)


    fig = plt.figure()

    sample = ds[8743]

    print(sample[1])
    plt.imshow(sample[0].permute(1,2,0))
    plt.show()


