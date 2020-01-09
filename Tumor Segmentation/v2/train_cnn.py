import torch, torch.utils.data, os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import CNN


### HYPERPARAMETERS
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 1
LOG_INTERVAL = 50
SHIP_DIRECTORY = os.getcwd() + "/data/ship_vae" #note: you must use the dataset of VAE contstructions..
WATER_DIRECTORY = os.getcwd() + "/data/water_vae" #note: you must use the dataset of VAE contstructions..

### LOAD DATA
ship = datasets.ImageFolder(root=SHIP_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

water = datasets.ImageFolder(root=WATER_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

ship_loader = torch.utils.data.DataLoader(dataset=ship, batch_size=BATCH_SIZE, shuffle=True)
water_loader = torch.utils.data.DataLoader(dataset=water, batch_size=BATCH_SIZE, shuffle=True)

### INIT MODEL
device = torch.device("cpu") #change if on GPU, also need to use .cuda()
model = CNN().to(device)

### MSE LOSS AND ADAM OPTIMIZER
criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

### TRAIN (SHIP == 1, WATER == 0)
for epoch in range(EPOCHS):

    print(f"EPOCH {epoch}")
    correct, total = 0, 0

    for idx, ((positive, _), (negative, __)) in enumerate(zip(ship_loader, water_loader)):
        
        #training on ship batch
        out_positive = model(positive)
        loss = criterion(out_positive, torch.ones(1))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if out_positive > 0.5:
            correct += 1

        #training on water batch
        out_negative = model(negative)
        loss = criterion(out_negative, torch.zeros(1))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if out_negative < 0.5:
            correct += 1

        total += 2

        if idx % LOG_INTERVAL == 0:
            print(f"Epoch: {epoch} [{idx * len(positive)} / {len(ship_loader.dataset)}] - Loss: {loss.item()}")
            print(f"ACCURACY: {100 * correct / total } % \n")

    torch.save(model, "saves/cnn.pt")


    

