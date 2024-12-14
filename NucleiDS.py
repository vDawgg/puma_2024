import torch
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from multiprocessing import Pool
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from wholeslidedata.annotation.labels import Label
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from monai.data import Dataset, DataLoader
import monai


class NucleiDS(Dataset):
    def __init__(self, image_path, geojson_path, image_transform, mask_transform=None):
        self.image_path = image_path
        self.image_file_names = os.listdir(image_path)
        self.image_transform = image_transform
        self.geojson_files = [file for file in os.listdir(geojson_path) if file.endswith(".geojson")]
        self.geojson_path = geojson_path
    def make_mask(self, geojson_path):
        tile_size = 1024
        class_map = {
            "nuclei_tumor": 1,
            "nuclei_lymphocyte": 2,
            "nuclei_plasma_cell": 2,
            "nuclei_histiocyte": 3,
            "nuclei_melanophage": 3,
            "nuclei_neutrophil": 3,
            "nuclei_stroma": 3,
            "nuclei_endothelium": 3,
            "nuclei_epithelium": 3,
            "nuclei_apoptosis": 3,
        }

        labels = [Label.create(label_name, value=label_value) for label_name, label_value in class_map.items()]
        wsa = WholeSlideAnnotation(geojson_path, labels=labels)

        shape = (1024, 1024)
        ratio = 1

        label_sampler = SegmentationPatchLabelSampler()
        for y_pos in range(0, shape[1], tile_size):
            for x_pos in range(0, shape[0], tile_size):
                mask = label_sampler.sample(
                    wsa,
                    (
                        (x_pos + tile_size // 2),
                        (y_pos + tile_size // 2),
                    ),
                    (tile_size, tile_size),
                    ratio,
                )
                return mask
    def __len__(self):
        return len(self.image_file_names)
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.image_file_names[index]))
        img = self.image_transform(image)
        mask = self.make_mask(os.path.join(self.geojson_path, self.geojson_files[index]))
        return img, mask

if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    dataset = NucleiDS(image_path="./data/01_training_dataset_tif_ROIs/", geojson_path="./data/01_training_dataset_geojson_nuclei/", image_transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(dataloader))
    print(images.size())
    print(masks.size())