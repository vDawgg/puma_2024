from typing import Any

from monai.data import DataLoader, decollate_batch, Dataset
from monai.optimizers import LearningRateFinder

import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(rounding="torchrounding")])

def train(train_loader: DataLoader, train_ds: Dataset, val_loader: DataLoader, model: Any, epochs: int, lr: float, loss_function) -> None:
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
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
                    val_outputs = model(val_images) #torch.argmax(model(val_images), dim=1, keepdim=True)
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
                    torch.save(model.state_dict(), "./models/best_metric_model_segmentation2d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

def find_lr(model, optimizer, l_f, data_loader) -> float:
    lr_finder = LearningRateFinder(model, optimizer, l_f)
    lr_finder.range_test(data_loader, end_lr=100, num_iter=100)
    lr, _ = lr_finder.get_steepest_gradient()
    lr_finder.plot()
    return lr