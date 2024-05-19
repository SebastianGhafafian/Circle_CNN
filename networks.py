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
        x = self.lay5(self.lay4(self.lay3(self.lay2(self.lay1(x)))))
        # get shape of resulting output
        B, C, H, W = x.shape
        # transform to tensor with shape infered from the remaining dimensions (C * H * W)
        x = x.view(-1, C * H * W) # results in tensor of shape [batchsize x (4*12*12=576)]
        x = self.fc(x)
        x = self.final(x)
        return x


class CustomCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        # set up convolutional layers conv -> batchnorm -> ReLu -> MaxPool
        self.lay1 = nn.Sequential(nn.Conv2d(1, 32, 5,padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 50 * 50
        self.lay2 = nn.Sequential(nn.Conv2d(32, 64, 3,padding = 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 25 * 25
        self.lay3 = nn.Sequential(nn.Conv2d(64, 128, 3,padding = 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 12 * 12
        self.lay4 = nn.Sequential(nn.Conv2d(128, 4, 1), nn.BatchNorm2d(4), nn.ReLU())  # 4 * 12 * 12
        # set up two linear  full connected layers with ReLu activation
        self.fc = nn.Sequential(nn.Linear(4 * 12*12, 48), nn.ReLU(),nn.Linear(48,16),nn.ReLU())
        # finalize with linear layers to output predictions x,y,r
        self.final = nn.Linear(16, 3)

    def forward(self, x):
        """Defines how data is fed forward through the network"""
        # apply convolutional layers
        x = self.lay4(self.lay3(self.lay2(self.lay1(x))))
        # get shape of resulting output
        B, C, H, W = x.shape
        # transform to tensor with shape infered from the remaining dimensions (C * H * W)
        x = x.view(-1, C * H * W) # results in tensor of shape [batchsize x (4*12*12=576)]
        x = self.FC(x)
        x = self.Last(x)
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
        x =  self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        # get shape of resulting output
        B, C, H, W = x.shape
        # transform to tensor with shape infered from the remaining dimensions (C * H * W)
        x = x.view(-1, C * H * W) # results in tensor of shape [batchsize x (4*12*12=576)]
        x = self.fc(x)
        x = self.final(x)
        return x

class LeNet(nn.Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super().__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=10,
			kernel_size=5,padding = 2)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = nn.Linear(in_features=400, out_features=100)
		self.relu3 = nn.ReLU()
		# initialize our softmax classifier
		self.fc2 = nn.Linear(in_features=100, out_features=classes)
		self.logSoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = nn.flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output

