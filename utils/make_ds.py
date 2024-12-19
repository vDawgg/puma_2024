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

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
nuclei_geojson_dir = os.path.join(data_dir, "01_training_dataset_geojson_nuclei")
tissue_geojson_dir = os.path.join(data_dir, "01_training_dataset_geojson_tissue")
ims_dir = os.path.join(data_dir, "01_training_dataset_tif_ROIs")
nuclei_masks_dir = os.path.join(data_dir, "masks_nuclei")
tissue_masks_dir = os.path.join(data_dir, "masks_tissue")


def make_mask(geojson_path: str, class_map: dict[str, int]) -> np.ndarray:
    tile_size = 1024

    labels = [Label.create(label_name, value=label_value) for label_name, label_value in class_map.items()]
    try:
        wsa = WholeSlideAnnotation(geojson_path, labels=labels)
    except Exception:
        print(labels)
        print(geojson_path)
        raise Exception

    ratio = 1

    label_sampler = SegmentationPatchLabelSampler()
    mask = label_sampler.sample(
        wsa,
        (
            tile_size // 2,
            tile_size // 2,
        ),
        (tile_size, tile_size),
        ratio,
    )
    return mask

def make_mask_nuclei(geojson_path):
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
    return make_mask(geojson_path, class_map)

def make_masks_tissue(geojson_path):
    class_map = {
        "tissue_stroma": 1,
        "tissue_blood_vessel": 2,
        "tissue_tumor": 3,
        "tissue_epidermis": 4,
        "tissue_necrosis": 5
    }
    return make_mask(geojson_path, class_map)

def test_masks(input_paths: [str], ppool: Any, output_path: str, map_func) -> [str]:
    file_names = [os.path.join(output_path, input_path.split("/")[-1].split(".")[0] + ".tif") for input_path in input_paths]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        for file_name, input_path in zip(file_names, input_paths):
            Image.fromarray(make_masks_tissue(input_path)).save(file_name)
    return file_names

def get_maks(input_paths: [str], ppool: Any, output_path: str, map_func) -> [str]:
    file_names = [os.path.join(output_path, input_path.split("/")[-1].split(".")[0] + ".tif") for input_path in input_paths]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        [Image.fromarray(mask).save(file_name) for mask, file_name in zip(ppool.map(map_func, input_paths), file_names)]

    return file_names

def get_ds(train_transforms: Compose,
           val_test_transforms: Compose,
           split=(0.8, 0.1, 0.1),
           task="nuclei") -> [Dataset]:
    if task == "nuclei":
        segs_path = nuclei_geojson_dir
        masks_path = nuclei_masks_dir
    else:
        segs_path = tissue_geojson_dir
        masks_path = tissue_masks_dir

    pool = Pool()
    metastatic_seg_paths = sorted([os.path.join(segs_path, seg_f) for seg_f in os.listdir(segs_path) if ('metastatic' in seg_f and ".geojson" in seg_f)])
    metastatic_img_paths = sorted([os.path.join(ims_dir, im_f) for im_f in os.listdir(ims_dir) if 'metastatic' in im_f])
    primary_seg_paths = sorted([os.path.join(segs_path, seg_f) for seg_f in os.listdir(segs_path) if ('primary' in seg_f and '.geojson' in seg_f)])
    primary_img_paths = sorted([os.path.join(ims_dir, im_f) for im_f in os.listdir(ims_dir) if 'primary' in im_f])

    train_split = int(len(metastatic_seg_paths)*split[0])
    val_split = int(len(metastatic_seg_paths)*split[1])

    train_img_paths = metastatic_img_paths[:train_split] + primary_img_paths[:train_split]
    train_seg_paths = metastatic_seg_paths[:train_split] + primary_seg_paths[:train_split]
    val_img_paths = metastatic_img_paths[train_split:val_split+train_split] + primary_img_paths[train_split:val_split+train_split]
    val_seg_paths = metastatic_seg_paths[train_split:val_split+train_split] + primary_seg_paths[train_split:val_split+train_split]
    test_img_paths = metastatic_img_paths[train_split+val_split:] + primary_img_paths[train_split+val_split:]
    test_seg_paths = metastatic_seg_paths[train_split+val_split:] + primary_seg_paths[train_split+val_split:]

    if task == "nuclei":
        train_mask_paths = get_maks(train_seg_paths, pool, os.path.join(masks_path, "train"), make_mask_nuclei)
        val_mask_paths = get_maks(val_seg_paths, pool, os.path.join(masks_path, "val"), make_mask_nuclei)
        test_mask_paths = get_maks(test_seg_paths, pool, os.path.join(masks_path, "test"), make_mask_nuclei)
    else:
        train_mask_paths = test_masks(train_seg_paths, pool, os.path.join(masks_path, "train"), make_mask_nuclei)
        val_mask_paths = test_masks(val_seg_paths, pool, os.path.join(masks_path, "val"), make_mask_nuclei)
        test_mask_paths = test_masks(test_seg_paths, pool, os.path.join(masks_path, "test"), make_mask_nuclei)

    train_ds = Dataset([{"image": img, "label": seg} for img, seg in zip(train_img_paths, train_mask_paths)], transform=train_transforms)
    val_ds = Dataset([{"image": img, "label": seg} for img, seg in zip(val_img_paths, val_mask_paths)], transform=val_test_transforms)
    test_ds = Dataset([{"image": img, "label": seg} for img, seg in zip(test_img_paths, test_mask_paths)], transform=val_test_transforms)
    
    return train_ds, val_ds, test_ds

def get_tissue_ds():
    pass


if __name__ == "__main__":
    base_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader=PILReader()),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
    )

    train_ds, val_ds, test_ds = get_ds(base_transforms, base_transforms, task="tissue")

    check_loader = DataLoader(val_ds)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["image"].shape, check_data["label"].shape)
    print(np.unique(check_data["label"]))

    plt.imshow(check_data["image"][0, 0, :, :])
    plt.show()
    plt.imshow(np.transpose(check_data["label"][0, :, :, :], (1, 2, 0)))
    plt.show()