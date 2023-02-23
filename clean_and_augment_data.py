import os
import json
from PIL import Image
import random

dataset_all = '/home/chiara/SEGMENTATION/DATASETS/DATASET_ASSIGNMENT1/'
base_path = os.path.dirname(os.path.abspath(__file__))

weird_scheds = ['573.png', '405.png', '575.png', '576.png', '1173.png', '593.png', '561.png', '572.png', '180.png', '1214.png', '2327.png', '571.png', '1480.png', '667.png', '174.png', '228.png', '2392.png', '1562.png', '408.png', '599.png', '1575.png', '210.png', '1479.png', '584.png', '586.png', '352.png', '579.png', '568.png', '583.png', '91.png', '798.png', '585.png', '1200.png', '562.png', '507.png', '577.png', '1189.png', '622.png', '1137.png', '581.png', '582.png', '170.png', '564.png']

occlusion_properties = ['occluded', 'occluding', 'occluded/occluding', 'neither']
categories = []
id = 0
for i in range(len(occlusion_properties)):
    name=occlusion_properties[i]
    category = {
        "supercategory": name,
        "name": name,
        "id": id
    }
    id = id + 1
    categories.append(category)

phases = ["train", "val", "test"]
for phase in phases:
    root_path = '/home/chiara/DATASETS/images/'
    gt_path = dataset_all+'annotations_per_image/'+phase+'/'
    json_file = base_path + '/dataset/instances_{}.json'.format(phase)

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annot_count = 0
    processed = 0
    image_id = 0
    allfiles = os.listdir(gt_path)
    for a in allfiles:
        file_path = os.path.join(gt_path, a)
        with open(file_path) as f:
            anns = json.load(f)
        key = list(anns.keys())[0]
        rep = 1
        file_name = anns[key]['filename']
        if file_name in weird_scheds:
            continue

        image = root_path + file_name
        width, height = Image.open(image).size
        if height != 720:
            continue

        img_elem = {"file_name": file_name,
                    "height": height,
                    "width": width,
                    "id": image_id}

        num_boxes = len(anns[key]['regions'])

        if num_boxes == 1:
            continue

        res_file["images"].append(img_elem)

        for i in range(num_boxes):
            xmin = int(anns[key]['regions'][i]['shape_attributes']['x'])
            ymin = int(anns[key]['regions'][i]['shape_attributes']['y'])
            w = int(anns[key]['regions'][i]['shape_attributes']['width'])
            h = int(anns[key]['regions'][i]['shape_attributes']['height'])
            bbox = [xmin, ymin, w, h]

            xmax = xmin + w
            ymax = ymin + h
            area = w * h

            scheduling = anns[key]['regions'][i]['region_attributes']['scheduling']
            occlusion = anns[key]['regions'][i]['region_attributes']['occlusion']

            category_id = 0
            for o in range(len(occlusion_properties)):
                if occlusion_properties[o] == occlusion:
                    category_id = o

            annot_elem = {
                "id": annot_count,
                "bbox": [
                    float(xmin),
                    float(ymin),
                    float(w),
                    float(h)
                ],
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": 0,
                "area": float(area),
                "caption": scheduling
            }

            res_file["annotations"].append(annot_elem)
            annot_count += 1

        image_id += 1

        processed += 1
    ''''''
    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")
