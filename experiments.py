'''
INPUT: raw image
OUTPUT: first strawberry to be picked, that will be succesfully picked

HOW?
1) image to DETR --> strawberry identification and classification
2) strawberries in graph1 representation
3) graph1 to scheduling prediction GAT model --> which is the target strawberry (as bbox coordinates in json file)
'''


import torch
import config as cfg
from detr.test import test_detr
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
base_path = os.path.dirname(os.path.abspath(__file__))


def first_bbox(first, json_path):
    with open(json_path + 'raw/gnnann.json') as f:
        gnnann = json.load(f)
    boxes = gnnann[0]['boxes']
    bbox = boxes[first]

    return bbox


def add_patch(img_path, bbox):
    orig_image = Image.open(img_path)
    img = np.array(orig_image)
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                               fill=True, color='white', linewidth=3))
    plt.axis('off')
    new_image_folder = img_path.strip(img_path.split('/')[-1]) + 'target'
    if not os.path.exists(new_image_folder):
        os.makedirs(new_image_folder)
    new_image = new_image_folder + '/' + img_path.split('/')[-1]
    plt.savefig(new_image)
    return new_image


def experiment(img_path):

    # obtain file with bounding boxes and occlusion properties from raw image
    json_annotations_path = test_detr(img_path)

    # obatin data in graph representation
    graph_data_scheduling = cfg.DATASET(json_annotations_path)

    # get first strawberry to pick
    best_GAT_scheduling_model = cfg.MODEL(cfg.HL, cfg.NL).to(device)
    best_GAT_scheduling_model.load_state_dict(
        torch.load(base_path + 'best_models/model_20230224_132115.pth'))
    scheduling_probability_vector = best_GAT_scheduling_model(graph_data_scheduling.get(0)).squeeze(1)
    first = scheduling_probability_vector.argmax()
    bbox = first_bbox(first, json_annotations_path)

    # obtain original image with white patch on target strawberry
    new_image_path = add_patch(img_path, bbox)

    return new_image_path


img_path = '/home/chiara/TRAJECTORIES/dataset_collection/dataset/strawberry_imgs/rgb_img_config0_strawberry0_traj0.png'
target_strawberry = experiment(img_path)
