import os
import json
from PIL import Image
import random

dataset_all = '/home/chiara/SEGMENTATION/DATASETS/DATASET_ASSIGNMENT1/'

occlusion_properties = ['occluded', 'occluding', 'occluded/occluding', 'neither']

# JUST OCCLUSION

phases = ["train", "val", "test"]
for phase in phases:
    root_path = '/home/chiara/DATASETS/images/'
    gt_path = dataset_all+'annotations_per_image/'+phase+'/'
    json_file = '/home/chiara/strawberry_picking_order_prediction/dataset/isamesize_{}.json'.format(phase)

    res_file = {
        "images": [],
        "annotations": []
    }

    annot_count = 0
    processed = 0
    image_id = 0
    allfiles = os.listdir(gt_path)
    for a in allfiles:
        for twice in range(4):
            file_path = os.path.join(gt_path, a)
            with open(file_path) as f:
                anns = json.load(f)
            key = list(anns.keys())[0]

            file_name = anns[key]['filename']

            image = root_path + file_name
            width, height = Image.open(image).size
            if height != 720:
                continue

            img_elem = {"file_name": str(twice)+'_'+file_name,
                        "height": height,
                        "width": width,
                        "id": image_id}

            num_boxes = len(anns[key]['regions'])

            rem = []
            if twice > 0:
                rem += [random.randint(0, num_boxes)]

            if num_boxes <= 1 + len(rem):
                continue

            res_file["images"].append(img_elem)

            for i in range(num_boxes):
                if i not in rem:
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
                        "scheduling": int(scheduling)
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
