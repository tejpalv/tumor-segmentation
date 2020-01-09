import torch, random, string, os
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE

BATCH_SIZE = 1
EPOCHS = 1
SHIP_DIRECTORY = os.getcwd() + "/data/ship_raw"
WATER_DIRECTORY = SHIP_DIRECTORY = os.getcwd() + "/data/water_raw"

ship = datasets.ImageFolder(root=SHIP_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
ship_loader = torch.utils.data.DataLoader(dataset=ship, batch_size=BATCH_SIZE, shuffle=True)

water = datasets.ImageFolder(root=WATER_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
water_loader = torch.utils.data.DataLoader(dataset=water, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cpu")
model = torch.load('vae.pt')

for epoch in range(EPOCHS):
    for idx, (sample, _1) in enumerate(ship_loader):
        if sample.shape[0] < BATCH_SIZE: break
        out = model(sample)

        g = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        save_image(out, f"data/ship_vae/train/1_{g}.png")

for epoch in range(EPOCHS):
    for idx, (sample, _1) in enumerate(water_loader):
        if sample.shape[0] < BATCH_SIZE: break
        out = model(sample)

        g = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        save_image(out, f"data/water_vae/0_{g}.png")



   