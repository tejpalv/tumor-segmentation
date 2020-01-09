import torch, random, string
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE

BATCH_SIZE = 1
EPOCHS = 1

POSITIVE_DIRECTORY = "/Users/john/Documents/morteza-reborn/data/water"

device = torch.device("cpu")

positive = datasets.ImageFolder(root=POSITIVE_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

positive_loader = torch.utils.data.DataLoader(dataset=positive, batch_size=BATCH_SIZE, shuffle=True)


model = torch.load('vae.pt')


for epoch in range(EPOCHS):
    for batch_index, (positive, _1) in enumerate(positive_loader):
        if positive.shape[0] < BATCH_SIZE: break
        out_positive = model(positive)

        g = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        save_image(out_positive, f"water/0_{g}.png")





torch.save(model, 'vae.pt')


   