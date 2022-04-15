import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    # Found this class template of medium and used it to make model
    def __init__(self, imgChannels=3, featureDim=51984, zDim=512):
        super(VAE, self).__init__()

        # Initializing the 7 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 8, 5)
        self.encConv2 = nn.Conv2d(8, 16, 5)
        self.encConv3 = nn.Conv2d(16, 16, 5)
        self.encConv4 = nn.Conv2d(16, 16, 4,stride=2)
        self.encConv5 = nn.Conv2d(16, 16, 4)
        self.encConv6 = nn.Conv2d(16, 16, 3)
        self.encConv7 = nn.Conv2d(16, 16, 3,stride=2)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 7 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(16, 16, 3,stride=2)
        self.decConv2 = nn.ConvTranspose2d(16, 16, 3)
        self.decConv3 = nn.ConvTranspose2d(16, 16, 4)
        self.decConv4 = nn.ConvTranspose2d(16, 16, 4,stride=2)
        self.decConv5 = nn.ConvTranspose2d(16, 16, 5)
        self.decConv6 = nn.ConvTranspose2d(16, 8, 5)
        self.decConv7 = nn.ConvTranspose2d(8, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = F.relu(self.encConv5(x))
        x = F.relu(self.encConv6(x))
        x = F.relu(self.encConv7(x))
        x = x.view(-1, 51984)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 16, 57, 57)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x=  F.relu(self.decConv3(x))
        x=  F.relu(self.decConv4(x))
        x=  F.relu(self.decConv5(x))
        x=  F.relu(self.decConv6(x))
        x = torch.sigmoid(self.decConv7(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

model=VAE()
print(model)