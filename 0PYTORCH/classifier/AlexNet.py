import pytorch_lightning as pl
import torch.nn as nn


# Same padding - 0 padding
# Valid padding - f//2 padding


class AlexNet(pl.LightningModule):

    def __init__(self, n_label, isTrain, verbose=True):
        super().__init__()
        dropout = 0.5 if isTrain else 1.0

        # Valid
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.norm1 = nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Same
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2)
        self.norm2 = nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=2)

        # Same
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        # Same
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2)

        # Same
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.fc6 = nn.Linear(in_features=4 * 4 * 256, out_features=4096)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout)

        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(dropout)

        self.fc8 = nn.Linear(in_features=4096, out_features=n_label)

        if verbose:
            print("Model", self, sep='\n')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = self.pool2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = x.view(-1, 4 * 4 * 256)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)

        x = self.fc8(x)

        return x
