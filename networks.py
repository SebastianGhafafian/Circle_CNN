import torch.nn as nn
class CustomCNN5(nn.Module):
    def __init__(self):
        super().__init__()
        # set up convolutional layers conv -> batchnorm -> ReLu -> MaxPool
        self.lay1 = nn.Sequential(nn.Conv2d(1, 32, 5,padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 50 * 50
        self.lay2 = nn.Sequential(nn.Conv2d(32, 64, 3,padding = 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 25 * 25
        self.lay3 = nn.Sequential(nn.Conv2d(64, 128, 3,padding = 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 12 * 12
        self.lay4 = nn.Sequential(nn.Conv2d(128, 128, 3,padding =1), nn.BatchNorm2d(128), nn.ReLU())  # 128 * 12 * 12
        self.lay5 = nn.Sequential(nn.Conv2d(128, 4, 1), nn.BatchNorm2d(4), nn.ReLU())  # 4 * 12 * 12
        # set up two linear  full connected layers with ReLu activation
        self.fc = nn.Sequential(nn.Linear(4 * 12*12, 48), nn.ReLU(),nn.Linear(48,16),nn.ReLU())
        # finalize with linear layers to output predictions x,y,r
        self.final = nn.Linear(16, 3)

    def forward(self, x):
        """Defines how data is fed forward through the network"""
        # apply convolutional layers
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        # get shape of resulting output
        B, C, H, W = x.shape
        # transform to tensor with shape infered from the remaining dimensions (C * H * W)
        x = x.view(-1, C * H * W) # results in tensor of shape [batchsize x (4*12*12=576)]
        x = self.fc(x)
        x = self.final(x)

        return x


class CustomCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        # set up convolutional layers conv -> batchnorm -> ReLu -> MaxPool
        self.lay1 = nn.Sequential(nn.Conv2d(1, 8, 5,padding=2), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2, 2))  # 8 * 50 * 50
        self.lay2 = nn.Sequential(nn.Conv2d(8, 16, 5,padding = 2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))  # 16 * 25 * 25
        self.lay3 = nn.Sequential(nn.Conv2d(16, 8, 3,padding = 1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2, 2))  # 8 * 12 * 12

        self.fc = nn.Sequential(nn.Linear(8 * 12 * 12, 16), nn.ReLU())
        # finalize with linear layers to display output variables
        self.final = nn.Linear(16, 3)

    def forward(self, x):
        """Defines how data is fed forward through the network"""
        # apply convolutional layers
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        # get shape of resulting output
        B, C, H, W = x.shape
        # transform to tensor with shape infered from the remaining dimensions (C * H * W)
        x = x.view(-1, C * H * W) # results in tensor of shape [batchsize x (4*12*12=576)]
        x = self.fc(x)
        x = self.final(x)

        return x


