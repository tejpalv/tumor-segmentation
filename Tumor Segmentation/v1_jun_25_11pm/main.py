#training the transformer, after training the vae and classifier already

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE, CNN, TRANSFORMER

BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-3
POSITIVE_DIRECTORY = "/Users/john/Documents/morteza-reborn/data/ship_vae"

device = torch.device("cpu")


positive = datasets.ImageFolder(root=POSITIVE_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))


positive_loader = torch.utils.data.DataLoader(dataset=positive, batch_size=1, shuffle=True)


cnn = torch.load("cnn.pt")
vae = torch.load("vae.pt")

def gv_margin(x, margin):
    return torch.max(margin, x)

def gv_loss(z_no_lesion, z_with_lesion):
    mse_loss = nn.MSELoss(reduce=True)
    nl = vae.decode(z_no_lesion)
    wl = vae.decode(z_with_lesion)
    gv = gv_margin(mse_loss(z_no_lesion, z_with_lesion), torch.tensor([1]).float())
    cnn_loss = 0.1 * cnn(nl)
    print(f"GV: {gv.data}  CNN: {cnn_loss.data}")
    loss =  gv + cnn_loss
    return loss, nl, wl

transform = TRANSFORMER().to(device)
criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(transform.parameters(), lr=LEARNING_RATE, amsgrad=True)



for (positive, _) in positive_loader:
    reconstruction, z_with_lesion = vae(positive) #taking a positive, extracting the recon and latent z
    for x in range(100):
        z_no_lesion = transform(z_with_lesion) #transforming latent z (lesion) => latent z (no lesion)
        loss, decoded_nl, decoded_wl = gv_loss(z_no_lesion, z_with_lesion)
        optimizer.zero_grad()
        loss.sum().backward(retain_graph=True)
        optimizer.step()
        
        print("LOSS:", loss.sum().data)

    negative = vae.decode(z_no_lesion)

    save_image(reconstruction, "groundtruth.png", )
    save_image(negative, "negative.png")
    exit()





