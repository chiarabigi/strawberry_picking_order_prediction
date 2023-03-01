import os
import json
import shutil
import csv
from PIL import Image

dataset_all = ''  # path/to/annotations
dst_path = ''  # path to save annotation

# define categories
categories=[]
category={
    "supercategory": 'fail',
    "name": 'fail',
    "id": 0
}
categories.append(category)
category={
    "supercategory": 'success',
    "name": 'success',
    "id": 1
}
categories.append(category)


# you need to have images and annotations split in train/val/test
phases = ['train', 'val', 'test']
for phase in phases:
    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }
    annot_count = 0
    image_id = 0
    processed = 0

    dataset_phase = dataset_all + phase
    allfiles=os.listdir(dataset_phase)
    for a in allfiles:
        file_path=os.path.join(dataset_phase,a)

        bbox_filepath = file_path + '/' + a + '_BBoxes.csv'
        with open(bbox_filepath) as csv_file:
            csvreader = csv.reader(csv_file)
            bbox = []
            for row in csvreader:
                bbox.append(row)
        bbox = bbox[-1]

        success_filepath = file_path + '/' + a + '_success.csv'
        with open(success_filepath) as csv_file:
            csvreader = csv.reader(csv_file)
            success = []
            for row in csvreader:
                success.append(row)
        success = int(float(success[-1][0]))

        image = file_path + '/' + a + '.png'
        width, height = Image.open(image).size
        dst = dst_path + phase + '/' + a + '.png'
        shutil.copy(image, dst)

        img_elem = {"file_name": a + '.png',
                    "height": height,
                    "width": width,
                    "id": image_id}

        res_file["images"].append(img_elem)


        xmin = float(bbox[0])
        ymin = float(bbox[1])
        w = float(bbox[2]) - float(bbox[0])
        h = float(bbox[3]) - float(bbox[1])
        bbox = [xmin, ymin, w, h]

        xmax = xmin + w
        ymax = ymin + h
        area = w * h
        poly = [[xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]]

        annot_elem = {
            "id": annot_count,
            "bbox": [
                float(xmin),
                float(ymin),
                float(w),
                float(h)
            ],
            #"segmentation": list([poly]),
            "image_id": image_id,
            #"ignore": 0,
            "category_id": success,
            "iscrowd": 0,
            "area": float(area)
        }

        res_file["annotations"].append(annot_elem)
        annot_count += 1

        image_id += 1

        processed += 1

    ''''''
    save_path = dst_path + 'annotations/instances_{}.json'.format(phase)
    with open(save_path, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")
