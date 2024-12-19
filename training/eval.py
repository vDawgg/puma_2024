import json
from typing import Any

import torch
from matplotlib import pyplot as plt
from monai.data import DataLoader

from utils.mask_to_json import convert_mask_to_json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_nuclei(test_loader: DataLoader, checkpoint_name: str, model: Any) -> None:
    model.load_state_dict(torch.load(f"models/{checkpoint_name}.pth"))
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
            output = torch.argmax(model(test_images), dim=1, keepdim=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

            fig.suptitle(test_data["label_file"])
            axes[0].imshow(output.cpu()[0, 0, :, :])
            axes[0].set_title('Output Image')
            axes[0].axis('off')
            axes[1].imshow(test_labels.cpu()[0, 0, :, :])
            axes[1].set_title('Test Image')
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

            json_out = convert_mask_to_json(output.cpu()[0, 0, :, :])

            with open("./data/output/nuclei/test.json", "w") as f:
                json.dump(json_out, f)