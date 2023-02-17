import json
import random
import statistics

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import statistics
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
#encoder = torch.load('/home/chiara/test_encoder.pth')  # autoencoder is to be downloaded

# unripe info
unripe_path = '/home/chiara/strawberry_picking_order_prediction/dataset/unripe.json'  # obtained with detectron2 ran on GPU
with open(unripe_path) as f:
    unripe_ann = json.load(f)


def overlap(bbox1, bbox2, diag):
    x1c = bbox1[0] + bbox1[2] / 2
    y1c = bbox1[1] + bbox1[3] / 2
    x2c = bbox2[0] + bbox2[2] / 2
    y2c = bbox2[1] + bbox2[3] / 2
    L = distance.euclidean((x1c, y1c), (x2c, y2c))
    if L < 0.5:
        return True
    else:
        return False


areas = []
boxes = 0
all_max_bbox = np.zeros(4)
phases=['train', 'val', 'test']
ripe = 0
unripe = 0
areas_images = []
images = 0
for phase in phases:
    print('new phase')
    filepath = '/home/chiara/strawberry_picking_order_prediction/dataset/scheduling/data_{}/raw/gnnann.json'.format(phase)
    gnnann = []
    max_bbox = np.zeros(4)
    json_path = '/home/chiara/strawberry_picking_order_prediction/dataset/instances_{}.json'.format(phase)
    img_path = '/home/chiara/DATASETS/images/'  # images are to be downloaded
    with open(json_path) as f:
        anns = json.load(f)
    img_ann=[]
    sc_ann=[]
    occ_ann=[]
    xy = []
    img_id = 0
    filename = anns['images'][img_id]['file_name']
    d = Image.open(img_path + filename)
    images += 1
    width, height = d.size
    areas_images.append(width*height)
    diag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))

    for i in range(len(anns['annotations'])):
        '''
        # statistics
        if anns['annotations'][i]['bbox'][0] > max_bbox[0]:
            max_bbox[0] = anns['annotations'][i]['bbox'][0] / diag
        if anns['annotations'][i]['bbox'][1] > max_bbox[1]:
            max_bbox[1] = anns['annotations'][i]['bbox'][1] / diag
        if anns['annotations'][i]['bbox'][2] > max_bbox[2]:
            max_bbox[2] = anns['annotations'][i]['bbox'][2] / diag
        if anns['annotations'][i]['bbox'][3] > max_bbox[3]:
            max_bbox[3] = anns['annotations'][i]['bbox'][3] / diag

        areas.append(anns['annotations'][i]['bbox'][2] * anns['annotations'][i]['bbox'][3])'''
        boxes += 1

        if anns['annotations'][i]['image_id']==img_id:
            img_ann.append([x / diag for x in anns['annotations'][i]['bbox']])
            sc_ann.append(int(anns['annotations'][i]['caption'].split(',')[-1]))
            occ_ann.append(anns['annotations'][i]['category_id'])
            bbox = anns['annotations'][i]['bbox']
            xy.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])
        if anns['annotations'][i]['image_id'] == img_id+1 or i == len(anns['annotations'])-1:
            # shuffle! or the model will learn first bbox = first to pick
            order = np.arange(0, len(img_ann))
            random.shuffle(order)
            img_ann = [img_ann[j] for j in order]
            sc_ann = [sc_ann[k] for k in order]
            occ_ann = [occ_ann[h] for h in order]
            xy = [xy[n] for n in order]

            unripe_info = [unripe_ann[ele] for ele in range(len(unripe_ann)) if
                           unripe_ann[ele]['file_name'] == filename]
            unripe_boxes = unripe_info[0]['bboxes']
            true_unripe = []
            if len(unripe_boxes) > 0:
                for idx, box in enumerate(unripe_boxes):
                    xmin = float(box[0])
                    ymin = float(box[1])
                    w = float(box[2] - box[0])
                    h = float(box[3] - box[1])
                    bbox = [xmin / diag, ymin / diag, w / diag, h / diag]
                    add = True
                    for el in range(len(img_ann)):
                        if overlap(bbox, img_ann[el], diag):
                            add = False
                    if add:
                        true_unripe.append(bbox)
                        xy.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])
                        if bbox[0] > max_bbox[0]:
                            max_bbox[0] = bbox[0]
                        if bbox[1] > max_bbox[1]:
                            max_bbox[1] = bbox[1]
                        if bbox[2] > max_bbox[2]:
                            max_bbox[2] = bbox[2]
                        if bbox[3] > max_bbox[3]:
                            max_bbox[3] = bbox[3]
                        areas.append(w * h)
                        boxes += 1

                if len(true_unripe) > 0:
                    occ_unripe = [3] * len(true_unripe)
                    sc_unripe = [18] * len(true_unripe)
                    occ_ann.extend(occ_unripe)
                    sc_ann.extend(sc_unripe)

            ''''''
            # add patches
            xy = torch.tensor(xy)
            x, y = xy.T
            d = transforms.ToTensor()(d)
            m = 112  # m = 1 means a patch of 3x3 centered around given pixel location
            for p in range(len(x)):
                if x[p] + m >= width:
                    x[p] = width - m
                elif x[p] - m <= 0:
                    x[p] = m
                if y[p] + m >= height:
                    y[p]= height - m
                elif y[p] - m <= 0:
                    y[p] = m
            o = torch.stack([d[:, yi - m: yi + m, xi - m: xi + m] for xi, yi in zip(x, y)])
            patches = []
            for a in range(len(o)):
                model.eval()
                compress = model.avgpool(model.features(o[a].unsqueeze(0))).squeeze(0).squeeze(-1).squeeze(-1)
                patches.append(compress.tolist())

            gnnann.append({'img_ann': img_ann,
                           'sc_ann': sc_ann,
                           'occ_ann': occ_ann,
                           # 'patches': patches,
                           'unripe': true_unripe,
                           'filename': filename
                           })
            ripe += len(img_ann)
            unripe += len(true_unripe)
            sc_ann = []
            img_ann = []
            occ_ann = []
            xy = []
            img_id = img_id + 1
            if img_id < len(anns['images']):
                filename = anns['images'][img_id]['file_name']
                d = Image.open(img_path + filename)
                images += 1
                width, height = d.size
                areas_images.append(width * height)
                diag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
                img_ann.append([x / diag for x in anns['annotations'][i]['bbox']])
                sc_ann.append(int(anns['annotations'][i]['caption'].split(',')[-1]))
                occ_ann.append(anns['annotations'][i]['category_id'])
                bbox = anns['annotations'][i]['bbox']
                xy.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])

    '''
    # statistics
    if max_bbox[0] > all_max_bbox[0]:
        all_max_bbox[0] = max_bbox[0]
    if max_bbox[1] > all_max_bbox[1]:
        all_max_bbox[1] = max_bbox[1]
    if max_bbox[2] > all_max_bbox[2]:
        all_max_bbox[2] = max_bbox[2]
    if max_bbox[3] > all_max_bbox[3]:
        all_max_bbox[3] = max_bbox[3]
    print(phase)
    print('max: ', max_bbox)'''


    ''''''
    # save
    with open(filepath, 'w') as f:
        json.dump(gnnann, f)
    print(phase + str(len(gnnann)))


''''''
# statistics
#print('together max: ', all_max_bbox)
print('max areas', max(areas))
print('min areas', min(areas))
print('avg area', sum(areas) / boxes)
print('median area', statistics.median(areas))
print('boxes', boxes)
print('max areas images', max(areas_images))
print('min areas images', min(areas_images))
print('avg area images', sum(areas_images) / images)
print('median area images', statistics.median(areas_images))
print('images', images)
print('ripe', ripe)
print('unripe', unripe)
