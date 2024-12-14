import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import Unet
from torch.utils.data import DataLoader
import torch.nn as nn
import monai
from make_ds import get_ds
from monai.data import PILReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd

# paths
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
geojson_dir = os.path.join(data_dir, "01_training_dataset_geojson_nuclei")
ims_dir = os.path.join(data_dir, "01_training_dataset_tif_ROIs")
masks_dir = os.path.join(data_dir, "masks_nuclei")

# parameter
n_epoch = 10
learning_rate = 1e-4
batch_size = 16
device = "cuda"

base_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"], reader=PILReader()),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
    ]
)

train_ds, val_ds, test_ds = get_ds(ims_dir, geojson_dir, masks_dir, base_transforms, base_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
sample = next(iter(train_loader))
print(f"Image size: {sample['img'].size()}")
print(f"Mask size: {sample['seg'].size()}")

#plt.imshow(sample_mask, cmap="viridis")
#plt.colorbar()
#plt.show()

net = Unet(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    channels=(4, 8, 16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=2
).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
training_loss_history = []
def training():
    net.train()
    for epoch in range(n_epoch):
        training_loss = 0
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images = batch["img"]
            masks = batch["seg"]
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            pred_mask = net(images)
            loss = loss_func(pred_mask, masks.long())
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(train_loader.dataset)
        print(f"Trainingloss {training_loss}")
        training_loss_history.append(training_loss)

    torch.save(net.state_dict(), "unet_model.pt")
    plt.plot(training_loss_history)
    plt.show()
    evaluation()
def evaluation():
    net.load_state_dict(torch.load("unet_model.pt", weights_only=True, map_location=device))
    net.eval()
    pred_mask = net(sample['img'].to(device))
    img_mask = torch.max(pred_mask, dim=1)[1]
    #print(torch.max(pred_mask, dim=1)[1].size())
    plt.imshow(img_mask[0].cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.show()

training()
#evaluation()