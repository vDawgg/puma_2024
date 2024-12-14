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
from monai.data import PILReader

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
geojson_dir = os.path.join(data_dir, "01_training_dataset_geojson_nuclei")
ims_dir = os.path.join(data_dir, "01_training_dataset_tif_ROIs")
masks_dir = os.path.join(data_dir, "masks_nuclei")

def make_mask(geojson_path):
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

def get_maks(input_paths: [str], ppool: Any, output_path: str) -> [str]:
    file_names = [os.path.join(output_path, input_path.split("/")[-1].split(".")[0] + ".tif") for input_path in input_paths]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        [Image.fromarray(mask).save(file_name) for mask, file_name in zip(ppool.map(make_mask, input_paths), file_names)]

    return file_names

def get_ds(ims_path: str,
           segs_path: str,
           masks_path: str,
           train_transforms: Compose,
           val_test_transforms: Compose,
           split=(0.8, 0.1, 0.1)) -> [Dataset]:
    pool = Pool()

    metastatic_seg_paths = sorted([os.path.join(segs_path, seg_f) for seg_f in os.listdir(segs_path) if 'metastatic' in seg_f])
    metastatic_img_paths = sorted([os.path.join(ims_path, im_f) for im_f in os.listdir(ims_path) if 'metastatic' in im_f])
    primary_seg_paths = sorted([os.path.join(segs_path, seg_f) for seg_f in os.listdir(segs_path) if 'primary' in seg_f])
    primary_img_paths = sorted([os.path.join(ims_path, im_f) for im_f in os.listdir(ims_path) if 'primary' in im_f])

    train_split = int(len(metastatic_seg_paths)*split[0])
    val_split = int(len(metastatic_seg_paths)*split[1])

    train_img_paths = metastatic_img_paths[:train_split] + primary_img_paths[:train_split]
    train_seg_paths = metastatic_seg_paths[:train_split] + primary_seg_paths[:train_split]
    val_img_paths = metastatic_img_paths[train_split:val_split] + primary_img_paths[train_split:val_split]
    val_seg_paths = metastatic_seg_paths[train_split:val_split] + primary_seg_paths[train_split:val_split]
    test_img_paths = metastatic_img_paths[train_split+val_split:] + primary_img_paths[train_split+val_split:]
    test_seg_paths = metastatic_seg_paths[train_split+val_split:] + primary_seg_paths[train_split+val_split:]

    train_mask_paths = get_maks(train_seg_paths, pool, os.path.join(masks_path, "train"))
    val_mask_paths = get_maks(val_seg_paths, pool, os.path.join(masks_path, "val"))
    test_mask_paths = get_maks(test_seg_paths, pool, os.path.join(masks_path, "test"))

    train_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(train_img_paths, train_mask_paths)], transform=train_transforms)
    val_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(val_img_paths, val_mask_paths)], transform=val_test_transforms)
    test_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(test_img_paths, test_mask_paths)], transform=val_test_transforms)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    base_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader=PILReader()),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )

    train_ds, val_ds, test_ds = get_ds(ims_dir, geojson_dir, masks_dir, base_transforms, base_transforms)

    check_loader = DataLoader(train_ds)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    plt.imshow(check_data["img"][0, 0, :, :])
    plt.show()
    plt.imshow(np.transpose(check_data["seg"][0, :, :, :], (1, 2, 0)))
    plt.show()