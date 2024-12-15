### The following file has been adapted from
### https://github.com/DIAGNijmegen/pathology-whole-slide-data/blob/main/scripts/convert_mask_to_xml_or_json.py

import json

from shapely import geometry
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import cv2
from PIL import Image
import numpy as np


def polygon_value_and_index_to_outer_inner(value, index, hierarchies_dict):
    """cv2_polygonize_with_hierarchy helper function: checks if poligon is outer (exterior) or inner (a hole) based on the hierarchy of the polygon"""
    hierarchy = hierarchies_dict[value][index]
    return 'outer' if hierarchy[3] == -1 else 'inner'


def polygons_dict_to_outer_polygons_dict(polygons_dict, hierarchies_dict, inv_label_map=None):
    """cv2_polygonize_with_hierarchy helper function: converts a dict of polygons to a dict of outer polygons, based on the hierarchy of the polygons. We typically only want to keep the outer polygons because WSD cannot read holes"""
    polygons_outer_dict = {}
    for value, polygons in polygons_dict.items():
        polygons_outer = [polygon for polygon_idx, polygon in enumerate(polygons) \
                          if polygon_value_and_index_to_outer_inner(value, polygon_idx, hierarchies_dict) == 'outer']
        label_name = f'class value {value}'
        if inv_label_map:
            label_name = inv_label_map[value]
        print(
            f'\t\t{label_name}: \n\t\t\tfrom {len(polygons)} polygons, {len(polygons_outer)} were outer, {len(polygons) - len(polygons_outer)} holes were removed')
        polygons_outer_dict[value] = polygons_outer
    return polygons_outer_dict


def cv2_polygonize_with_hierarchy(
        mask, dilation_iterations=0, erose_iterations=0, exclude_holes=True, values=None, inv_label_map=None):
    """converts a mask to a dict of polygons, with the option to exclude holes, based on 2 step hierarchy (exteriors and holes)."""
    if values is None:
        values = np.unique(mask)

    all_polygons = {}
    all_hierarchies = {}

    print('\tExtracting polygons with exterior/hole hierachy')
    for value in values:
        print(f'\t\tprocessing value {value}{f", {inv_label_map[value]}" if inv_label_map else ""}')

        tmp_mask = (mask == value).astype(
            np.uint8)  # improved here, allowing to extraxt background polygons (if you dont want background, exclude its value from 'values' input)

        if dilation_iterations > 0:
            tmp_mask = binary_dilation(tmp_mask, iterations=dilation_iterations).astype(
                np.uint8
            )
        if erose_iterations > 0:
            tmp_mask = binary_erosion(tmp_mask, iterations=erose_iterations).astype(
                np.uint8
            )

        tmp_mask = np.pad(
            array=tmp_mask, pad_width=1, mode="constant", constant_values=0
        )

        polygons, hierarchies = cv2.findContours(
            tmp_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE, offset=(-1, -1)
            # cv2.RETR_CCOMP retuns hierarchy with outer/hole inforamtion
        )

        if len(polygons) == 0:
            all_polygons[value] = []
            all_hierarchies[value] = []
            continue

        # remove instances with <3 coordinates
        filtered_polygonsand_hierarchies = [(np.array(polygon[:, 0, :]), hierarchy)
                                            for polygon, hierarchy in zip(polygons, hierarchies[0])
                                            if len(polygon) >= 3]
        if len(filtered_polygonsand_hierarchies) == 0:
            all_polygons[value] = []
            all_hierarchies[value] = []
            continue
        polygons, hierarchies = zip(*filtered_polygonsand_hierarchies)

        all_polygons[value] = polygons
        all_hierarchies[value] = hierarchies

    if exclude_holes:
        print('\tRemoving hole polygons')
        all_polygons = polygons_dict_to_outer_polygons_dict(all_polygons, all_hierarchies, inv_label_map)

    return all_polygons

def convert_polygons_to_json(polygons, inv_label_map):
    json_output = {
        "type": "Multiple polygons",
        "polygons": []
    }
    for value, polygons in polygons.items():
        for polygon in polygons:
            if isinstance(polygon, geometry.MultiPolygon):
                for q in list(polygon.geoms):
                    json_output["polygons"].append({
                        "name": inv_label_map[value],
                        "seed_point": list(q.exterior.coords[0]).__add__([0.5]),
                        "path_points": [list(x).__add__([0.5]) for x in q.exterior.coords],
                        "sub_type": "",
                        "groups": [],
                        "probability": 1,
                    })
            else:
                json_output["polygons"].append({
                    "name": inv_label_map[value],
                    "seed_point": list(polygon.exterior.coords[0]).__add__([0.5]),
                    "path_points": [list(x).__add__([0.5]) for x in polygon.exterior.coords],
                    "sub_type": "",
                    "groups": [],
                    "probability": 1,
                })
    return json_output

def convert_mask_to_xml_or_json(
        mask,
        label_mapping: {},
        dilation_iterations: int = 0,
        erose_iterations: int = 0,
        exclude_holes: bool = True, # TODO: Check whether this really needs to be true!
):
    inv_label_map = {value: name.lower() for value, name in label_mapping.items()}

    polygons = cv2_polygonize_with_hierarchy(
        mask,
        dilation_iterations=dilation_iterations,
        erose_iterations=erose_iterations,
        exclude_holes=exclude_holes,
        values=list(inv_label_map),
        inv_label_map=inv_label_map
    )

    for value, polys in polygons.items():
        for poly_idx, poly in enumerate(polys):
            polygons[value][poly_idx] = geometry.Polygon(poly)

    return convert_polygons_to_json(polygons, inv_label_map)

def convert_mask_to_json(mask):
    labels = {
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
    label_mapping = {v:k for k,v in labels.items()}

    return convert_mask_to_xml_or_json(
        mask=mask,
        label_mapping=label_mapping,
    )

if __name__ == "__main__":
    mask = np.array(Image.open('../data/masks_nuclei/train/training_set_primary_roi_001_nuclei.tif'))
    json_output = convert_mask_to_json(mask)
    print(json_output)
    with open("test.json", "w") as f:
        json.dump(json_output, f)