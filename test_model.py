import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import Unet
from torch.utils.data import DataLoader
from NucleiDS import NucleiDS
from torchvision.transforms import v2

transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
dataset = NucleiDS(image_path="./data/01_training_dataset_tif_ROIs/", geojson_path="./data/01_training_dataset_geojson_nuclei/", image_transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
images, masks = next(iter(dataloader))
print(images.size())

sample_img = images[0].numpy().transpose((1, 2, 0))
sample_mask = masks[0].numpy()

#plt.imshow(sample_mask, cmap="viridis")
#plt.colorbar()
#plt.show()

net = Unet(
    spatial_dims=2,
    in_channels=4,
    out_channels=1,
    channels=(4, 8, 16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=2
)

pred_mask = net(images)
print(pred_mask.size())