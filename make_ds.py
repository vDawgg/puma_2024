from multiprocessing import Pool
import os

import torch
from PIL import Image
from wholeslidedata.annotation.labels import Label
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from monai.data import Dataset

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

def get_ds(ims_path: str, segs_path: str, out_path: str, split=(0.8, 0.1, 0.1)) -> [Dataset]:
    if not os.path.exists(os.path.join(out_path, "train.pt")):
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

        train_masks = pool.map(make_mask, train_seg_paths)
        train_imgs = pool.map(Image.open, train_img_paths)
        val_masks = pool.map(make_mask, val_seg_paths)
        val_imgs = pool.map(Image.open, val_img_paths)
        test_masks = pool.map(make_mask, test_seg_paths)
        test_imgs = pool.map(Image.open, test_img_paths)

        train_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(train_imgs, train_masks)])
        val_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(val_imgs, val_masks)])
        test_ds = Dataset([{"img": img, "seg": seg} for img, seg in zip(test_imgs, test_masks)])

        torch.save(train_ds, os.path.join(out_path, "train_ds.pt"))
        torch.save(val_ds, os.path.join(out_path, "val_ds.pt"))
        torch.save(test_ds, os.path.join(out_path, "test_ds.pt"))

    else:
        train_ds = torch.load(os.path.join(out_path, "train_ds.pt"))
        val_ds = torch.load(os.path.join(out_path, "val_ds.pt"))
        test_ds = torch.load(os.path.join(out_path, "test_ds.pt"))

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    ims_path = "./data/01_training_dataset_tif_ROIs"
    segs_path = "./data/01_training_dataset_geojson_nuclei"
    out_path = "./data/output"

    train_ds, val_ds, test_ds = get_ds(ims_path, segs_path, out_path)