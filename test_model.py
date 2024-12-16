import torch
import os
import matplotlib.pyplot as plt
from monai.networks.nets import Unet, AttentionUnet
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_ds import get_ds
from monai.data import PILReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandFlipd, RandRotated, RandZoomd, RandSpatialCropd
from torch.amp import autocast, GradScaler

# paths
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
geojson_dir = os.path.join(data_dir, "01_training_dataset_geojson_nuclei")
ims_dir = os.path.join(data_dir, "01_training_dataset_tif_ROIs")
masks_dir = os.path.join(data_dir, "masks_nuclei")

# parameter
n_epoch = 100
learning_rate = 1e-5
batch_size = 32
device = "cuda"

base_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"], reader=PILReader()),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
        RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=1),
        RandRotated(keys=["img", "seg"], range_x=15, prob=0.5, keep_size=True),
        RandZoomd(keys=["img", "seg"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        RandSpatialCropd(keys=["img", "seg"], roi_size=(256, 256), random_size=False)
    ]
)

train_ds, val_ds, test_ds = get_ds(ims_dir, geojson_dir, masks_dir, base_transforms, base_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)
val_loader = DataLoader(val_ds, batch_size=batch_size)

print(f"Größe Trainingsdaten {len(train_loader.dataset)}, Größe Validationsdaten {len(val_loader.dataset)}, Größe Testdaten {len(test_loader.dataset)}")

sample = next(iter(train_loader))
def plot_sample():
    sample_mask = sample["seg"][0].numpy().transpose((1, 2, 0))
    print(f"Image size: {sample['img'].size()}")
    print(f"Mask size: {sample['seg'].size()}")

    sample_img = sample["img"][0].numpy().transpose((1, 2, 0))
    plt.imshow(sample_img)
    plt.imshow(sample_mask, cmap="viridis", alpha=0.2)
    plt.show()

'''
net = Unet(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 64, 128, 128, 256, 256),
    strides=(2, 2, 2, 2, 2, 2, 2),
    num_res_units=2
).to(device)
'''
net = AttentionUnet(spatial_dims=2,
                    in_channels=4,
                    out_channels=4,
                    channels=(16, 32, 64, 128, 128, 256, 512),
                    strides=(2, 2, 2, 2, 2, 2),
                    kernel_size=5,
                    up_kernel_size=5,
                    dropout=0.2).to(device)#'''
# oder SegResNet
net.load_state_dict(torch.load("unet_model.pt", weights_only=True, map_location=device))

loss_func = nn.CrossEntropyLoss() # ganz wichtig, Dice Loss implementieren. Klassen sind stark unbalanciert.
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
# man könnte noch ReduceLROnPlateau hinzufügen, lohnt sich momentan aber noch nicht
scaler = GradScaler()
training_loss_history = []
test_loss_history = []
def training():
    for epoch in range(n_epoch):
        # training
        net.train()
        training_loss = 0
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images = batch["img"]
            masks = batch["seg"]
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            with autocast(device_type="cuda"):
                pred_mask = net(images)
                loss = loss_func(pred_mask, masks.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()
            training_loss += loss.item()
        # testing
        test_loss = 0
        net.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                images = batch["img"]
                masks = batch["seg"]
                images = images.to(device)
                masks = masks.to(device).squeeze(1)
                pred_mask = net(images)
                loss = loss_func(pred_mask, masks.long())
                test_loss += loss.item()
        training_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print(f"Epoche: {epoch}, Trainingloss: {training_loss}, Testloss: {test_loss}")
        training_loss_history.append(training_loss)
        test_loss_history.append(test_loss)
        if len(test_loss_history) >= 2:
            if test_loss_history[-1] < test_loss_history[-2]: # immer speichern, wenn die testloss besser geworden ist
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