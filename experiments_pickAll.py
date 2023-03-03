'''
INPUT: raw image
OUTPUT: first strawberry to be picked

HOW?
1) image to DETR --> strawberry identification and classification
2) strawberries in graph1 representation
3) graph1 to scheduling prediction GAT model --> which is the target strawberry
'''


import torch
import config as cfg
from detr.test import test_detr
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
# from detectron2.inference_dyson_keypoints import test_detectron2
from data_scripts.detr_to_gnnann import ann_to_gnnann

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
base_path = os.path.dirname(os.path.abspath(__file__))


def get_ripe(json_path):
    with open(json_path + 'raw/gnnann.json') as f:
        gnnann = json.load(f)
    ripeness = gnnann[0]['ripeness']
    ripes = sum(1 for r in ripeness if r == 1)

    return ripes


def first_bbox(idx, json_path):
    with open(json_path + 'raw/gnnann.json') as f:
        gnnann = json.load(f)
    boxes = gnnann[0]['boxes']
    bbox = boxes[idx]

    return bbox


def add_patch(img_path, bbox, i, new_image_folder):
    orig_image = Image.open(img_path)
    draw = ImageDraw.Draw(orig_image)
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])], outline='white', fill='white')
    #w, h = orig_image.size
    #img = np.array(orig_image)
    #plt.figure(figsize=(w/100,h/100))
    #plt.imshow(img)
    #ax = plt.gca()
    #ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=True, color='white', linewidth=3))
    #plt.axis('off')
    new_image = new_image_folder + '/' + str(i + 1) + '_' + img_path.split('/')[-1]
    #plt.tight_layout()
    #plt.savefig(new_image, facecolor='black')
    orig_image.save(new_image)
    return new_image


def experiment(image_path, exp):

    # obtain bounding boxes of unripe strawberries from raw image
    unripe_info = [{
            'file_name': [],
            'bboxes': []
        }]  # test_detectron2(image_path, save=False)

    # obtain bounding boxes and occlusion properties from raw image
    occlusion_info = test_detr(image_path)

    json_annotations_path = ann_to_gnnann(occlusion_info, unripe_info, image_path)

    # obatin data in graph representation
    graph_data_scheduling = cfg.DATASET(json_annotations_path)

    # get scheduling
    best_GAT_scheduling_model = cfg.MODEL(cfg.HL, cfg.NL).to(device)
    best_GAT_scheduling_model.load_state_dict(
        torch.load(base_path + '/best_models/model_patches.pth', map_location=device))
    scheduling_probability_vector = best_GAT_scheduling_model(graph_data_scheduling.get(0)).squeeze(1)
    sched = sorted(range(len(scheduling_probability_vector)), reverse=True, key=lambda k: scheduling_probability_vector[k])

    # get images to pick ripe strawberries in order
    new_image_folder = image_path.strip(image_path.split('/')[-1]) + 'target{}'.format(exp)
    if not os.path.exists(new_image_folder):
        os.makedirs(new_image_folder)

    ripes = get_ripe(json_annotations_path)
    for i in range(ripes):
        idx = sched.index(i)
        bbox = first_bbox(idx, json_annotations_path)
        # obtain original image with white patch on target strawberry
        new_image_path = add_patch(image_path, bbox, i, new_image_folder)

    return new_image_folder


exp = 1
image_path = '/home/chiara/riseholme-experiments/pickall/{}/test{}_Color_Color.png'.format(exp, exp)
target_strawberries_folder = experiment(image_path, exp)
