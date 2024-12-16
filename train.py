from typing import Any

from PIL import Image
from matplotlib import pyplot as plt
from monai.data import DataLoader, list_data_collate, decollate_batch

from utils.make_ds import get_ds, ims_dir, geojson_dir, masks_dir
import monai
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityd

from utils.mask_to_json import convert_mask_to_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def train(train_loader: DataLoader, val_loader: DataLoader, model: Any, epochs: int) -> None:
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

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
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
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
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
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
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
            """roi_size = (96, 96)
            sw_batch_size = 4
            test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
            test_outputs = [post_trans(i).cpu() for i in decollate_batch(test_outputs)]"""
            output = model(test_images)
            img_mask = torch.max(output, dim=1)[1]
            # print(torch.max(pred_mask, dim=1)[1].size())
            plt.imshow(img_mask[0].cpu().numpy(), cmap="viridis")
            plt.colorbar()
            plt.show()
            #json = convert_mask_to_json(output[0])

if __name__ == "__main__":
    base_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )
    train_ds, val_ds, test_ds = get_ds(ims_dir, geojson_dir, masks_dir, base_transforms, base_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    """train(
        train_loader,
        val_loader,
        model,
        100000
    )"""

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    evaluate(test_loader, "unet_no_transforms_80-10-10_split", model)