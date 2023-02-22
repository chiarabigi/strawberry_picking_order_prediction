import os
import json
from PIL import Image
import random

dataset_all = '/home/chiara/SEGMENTATION/DATASETS/DATASET_ASSIGNMENT1/'

occlusion_properties = ['occluded', 'occluding', 'occluded/occluding', 'neither']

# JUST OCCLUSION
# big_scores = ['424.png', '442.png', '627.png', '573.png', '2390.png', '500.png', '524.png', '790.png', '1585.png', '366.png', '1569.png', '609.png', '473.png', '204.png', '427.png', '1575.png', '1120.png', '1543.png', '279.png', '458.png', '512.png', '118.png', '586.png', '437.png', '568.png', '555.png', '402.png', '798.png', '506.png', '336.png', '1658.png', '675.png', '1508.png', '444.png', '481.png', '254.png', '1524.png', '523.png']
big_scores = []
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
        file_path = os.path.join(gt_path, a)
        with open(file_path) as f:
            anns = json.load(f)
        key = list(anns.keys())[0]
        rep = 1
        file_name = anns[key]['filename']
        if file_name == '579.png':
            print(0)
        image = root_path + file_name
        width, height = Image.open(image).size
        if height != 720:
            continue

        if file_name in big_scores:
            rep = 100
        for twice in range(rep):

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
                    xmin = int(anns[key]['regions'][i]['shape_attributes']['x']) + twice * 3
                    ymin = int(anns[key]['regions'][i]['shape_attributes']['y']) + twice * 3
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
    '''
    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)'''

    print("Processed {} {} images...".format(processed, phase))
print("Done.")
