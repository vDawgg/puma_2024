from typing import Any

from matplotlib import pyplot as plt
from monai.data import DataLoader, list_data_collate, decollate_batch, PILReader
from monai.networks.nets import AttentionUnet
from monai.optimizers import LearningRateFinder

from utils.make_ds import get_ds, ims_dir, geojson_dir, masks_dir
import monai
import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, \
    RandFlipd, RandRotated, RandZoomd, RandSpatialCropd

from utils.mask_to_json import convert_mask_to_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(rounding="torchrounding")])

def train(train_loader: DataLoader, val_loader: DataLoader, model: Any, epochs: int, lr: float) -> None:
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start a typical PyTorch training
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_images)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "models/best_metric_model_segmentation2d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


def evaluate(test_loader: DataLoader, checkpoint_name: str, model: Any) -> None:
    model.load_state_dict(torch.load(f"models/{checkpoint_name}.pth"))
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
            output = model(test_images)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

            # Display the first image in the first subplot
            axes[0].imshow(output.cpu()[0, 0, :, :])
            axes[0].set_title('Output Image')
            axes[0].axis('off')  # Hide axis if desired

            # Display the second image in the second subplot
            axes[1].imshow(test_images.cpu()[0, 0, :, :])
            axes[1].set_title('Test Image')
            axes[1].axis('off')  # Hide axis if desired

            plt.tight_layout()  # Adjust spacing between subplots
            plt.show()
            #json = convert_mask_to_json(output[0])

def find_lr(model, optimizer, l_f, data_loader) -> float:
    lr_finder = LearningRateFinder(model, optimizer, l_f)
    lr_finder.range_test(data_loader, end_lr=100, num_iter=100)
    lr, _ = lr_finder.get_steepest_gradient()
    lr_finder.plot()
    return lr


if __name__ == "__main__":
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader=PILReader()),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
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
            ScaleIntensityd(keys=["image", "label"]),
        ]
    )
    train_ds, val_ds, test_ds = get_ds(ims_dir, geojson_dir, masks_dir, train_transforms, val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 128, 256, 512),
        strides=(2, 2, 2, 2, 2, 2),
        kernel_size=5,
        up_kernel_size=5,
        dropout=0.2
    ).to(device)

    lr = find_lr(model, torch.optim.Adam(model.parameters()), monai.losses.DiceLoss(sigmoid=True), train_loader)

    train(
        train_loader,
        val_loader,
        model,
        1000,
        lr
    )

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    evaluate(test_loader, "best_metric_model_segmentation2d_dict", model)