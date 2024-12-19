import monai
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.data import PILReader, DataLoader, list_data_collate
from monai.networks.nets import AttentionUnet, SwinUNETR
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandFlipd, RandRotated, \
    RandZoomd, RandSpatialCropd

from training.train import find_lr, train
from training.eval import evaluate_nuclei
from utils.make_ds import get_ds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=PILReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotated(keys=["image", "label"], range_x=15, prob=0.5, keep_size=True),
        RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=(256, 256), random_size=False)
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=PILReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
    ]
)
train_ds, val_ds, test_ds = get_ds(train_transforms, val_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

model = SwinUNETR(img_size=(1024, 1024), in_channels=4, out_channels=4, use_checkpoint=True, spatial_dims=2, use_v2=True).to(device)

"""model = AttentionUnet(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 128, 256, 512),
    strides=(2, 2, 2, 2, 2, 2),
    kernel_size=5,
    up_kernel_size=5,
    dropout=0.2
).to(device)"""

loss_function = (monai.losses.DiceFocalLoss(include_background=False, sigmoid=True, to_onehot_y=True))

lr = find_lr(model,
             torch.optim.Adam(model.parameters()),
             loss_function,
             train_loader)

train(
    train_loader,
    train_ds,
    val_loader,
    model,
    1000,
    lr,
    loss_function,
)

test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
evaluate_nuclei(test_loader, "best_metric_model_segmentation2d_dict", model)