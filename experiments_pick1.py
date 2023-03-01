'''
INPUT: raw image
OUTPUT: first strawberry to be picked

HOW?
1) image to DETR --> strawberry identification and classification
2) strawberries in graph1 representation
3) graph1 to scheduling prediction GAT model --> which is the target strawberry
'''

# WE USE AN OLD MODEL, THAT WAS BETTER AT PREDICTING THE FIRST STRAWBERRY TO BE PICKED

import torch
from model import GCN_OLD_scheduling
from data_scripts.old_dataset import SchedulingDataset
from detr.test import test_detr
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# from detectron2.inference_dyson_keypoints import test_detectron2
from data_scripts.detr_to_gnnann import old_ann_to_gnnann

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
base_path = os.path.dirname(os.path.abspath(__file__))


def first_bbox(idx, json_path):
    with open(json_path + 'raw/gnnann.json') as f:
        gnnann = json.load(f)
    boxes = gnnann[0]['img_ann']
    bbox = boxes[idx]

    return bbox


def add_patch(img_path, bbox, i):
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
    new_image = new_image_folder + '/' + str(i + 1) + '_' + img_path.split('/')[-1]
    plt.savefig(new_image)
    return new_image


def experiment(image_path):

    # obtain bounding boxes and occlusion properties from raw image
    occlusion_info = test_detr(image_path)

    json_annotations_path = old_ann_to_gnnann(occlusion_info, image_path)

    # obatin data in graph representation
    graph_data_scheduling = SchedulingDataset(json_annotations_path)

    # get scheduling
    model = GCN_OLD_scheduling
    best_GAT_scheduling_model = model(8, 0).to(device)
    best_GAT_scheduling_model.load_state_dict(
        torch.load(base_path + '/best_models/model_best_sched.pth', map_location=device))
    scheduling_probability_vector = best_GAT_scheduling_model(graph_data_scheduling.get(0)).squeeze(1)

    # get images to pick ripe strawberries in order
    idx = scheduling_probability_vector.argmax()
    bbox = first_bbox(idx, json_annotations_path)
    # obtain original image with white patch on target strawberry
    new_image_path = add_patch(image_path, bbox, 1)

    new_images_folder = image_path.strip(image_path.split('/')[-1]) + 'target'
    return new_images_folder


image_path = '/home/chiara/TRAJECTORIES/dataset_collection/dataset/strawberry_imgs/rgb_img_config0_strawberry0_traj0.png'
target_strawberries_folder = experiment(image_path)
