import torch, torch.utils.data, os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE

### HYPERPARAMETERS
BATCH_SIZE = 5
LEARNING_RATE = 1e-3 
EPOCHS = 31
LOG_INTERVAL = 50
DATA_DIRECTORY = os.getcwd() + "/data/both_raw_balanced"
SAVE_IMAGES_IN_RESULTS = True

### LOAD DATA
data = datasets.ImageFolder(root=DATA_DIRECTORY, transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                               ]))

data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

### INIT MODEL
device = torch.device("cpu") #change if on GPU, also need to use .cuda()
model = AE().to(device)

### MSE LOSS AND ADAM OPTIMIZER
criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

### TRAIN
for epoch in range(EPOCHS):
    for idx, (sample, _) in enumerate(data_loader):
        if sample.shape[0] < BATCH_SIZE: break
        reconstruction, z = model(sample)
        loss = criterion(reconstruction, sample)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if idx % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{idx * len(sample)} / {len(data_loader.dataset)}] - Loss: {loss.sum()}")

            if SAVE_IMAGES_IN_RESULTS:
                n = min(sample.size(0), 8)
                comparison = torch.cat([sample[:n], reconstruction.view(BATCH_SIZE, 3, 80, 80)[:n]])
                save_image(comparison.cpu(), f"results/VAE_{str(epoch)}_{str(idx)}.png", nrow=n)

### SAVING MODEL AT END
torch.save(model, 'saves/vae.pt')


   