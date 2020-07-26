import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes):
        super(CNN, self).__init__()

        # Get feature extracor
        self.model = feature_extractor

        for param in self.model.parameters():
            param.requires_grad = False

        n_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
