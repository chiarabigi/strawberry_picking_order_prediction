#! /home/rick/anaconda3/envs/fruitcast/bin/python3
# import some common detectron2 utilities
import json
import os
import pathlib
import random
import re
import sys

import cv2
import joblib
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.data import MetadataCatalog, DatasetCatalog

# sys.path.insert(1, '/data/lincoln/weight_estimation')
#
# import model_prediction_realsense
#
# from model_prediction_realsense import predict_weight

working_dir = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm

# from strawberry_keypoints import create_dump

# working_dir = './'


def read_json(file_path):
    with open(file_path, 'r') as read_file:
        data = json.load(read_file)
    read_file.close()
    return data


for d in ["train"]:
    DatasetCatalog.register("strawberry_" + d, lambda d=d: read_json(working_dir + d + '.json'))
    MetadataCatalog.get("strawberry_" + d).set(thing_classes=["pluckable", "unpluckable"])
    MetadataCatalog.get("strawberry_" + d).set(keypoint_names=["pluck", "top", "bottom", "left", "right"])
    MetadataCatalog.get("strawberry_" + d).set(keypoint_flip_map=[],)
    MetadataCatalog.get("strawberry_" + d).set(keypoint_connection_rules=[("pluck", "top", (0, 255, 255)),
                                                                          ("top", "bottom", (255, 255, 0)),
                                                                          ("left", "right", (0, 0, 255))])
strawberry_metadata = MetadataCatalog.get("strawberry_train")

# strawberry_metadata = MetadataCatalog.get("strawberry_train")


cfg = get_cfg()
cfg.OUTPUT_DIR = working_dir + '/checkpoint'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("strawberry_train",)
cfg.DATASETS.TEST = ('strawberry_dyson_lincoln_tbd__031_1_rgb.png')
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1100
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (strawberry).
# cfg.MODEL.KEYPOINT_ON = True
# cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 0
# cfg.TEST.AUG.FLIP = False
# NOTE: this config means the number of classes, but a few popular tutorials incorrect uses num_classes+1 here.

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.KEYPOINT_ON = True
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
# cfg.TEST.AUG.FLIP = False
# cfg.INPUT.RANDOM_FLIP = 'vertical'
predictor = DefaultPredictor(cfg)
print(cfg)

# img_dir = './'
unann = []

img_dir = ''  # path to images


def test_detectron2(img_dir, save):

    dataset_dicts = []
    # file_name_list = []

    for idx, rgb_file in tqdm(enumerate(pathlib.Path(img_dir).rglob('*.png'))):

        # print('org_file: ', rgb_file.as_posix())
        im = cv2.imread(rgb_file.as_posix())
        json_name = re.sub('.png', '.json', rgb_file.name)
        # im = cv2.imread(file_name)
        # print(im.shape)

        # cv2.imshow('inference', im)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        # print('outputs: ', outputs['instances'].to('cpu'))

        ''''''
        # save unripe boxes info
        outputs = outputs['instances'].to('cpu')
        outputsF = outputs.get_fields()
        all_boxes = outputsF['pred_boxes'].tensor.tolist()
        classes = outputsF['pred_classes'].tolist()
        boxes = []
        for i in range(len(all_boxes)):
            if classes[i] != 0:
                boxes.append(all_boxes[i])
        unripe_ann = {
            'file_name': rgb_file.name,
            'bboxes': boxes
        }
        unann.append(unripe_ann)

        '''
        # to visualise
        print('image file', rgb_file.as_posix())
        # print('outputs: ', outputs)
        v = Visualizer(im[:, :, ::-1],
                       metadata=strawberry_metadata,
                       scale=1,
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       instance_mode=ColorMode.IMAGE_BW
                       )
    
        out = v.draw_instance_predictions(outputs)
        cv2.imshow('inference', out.get_image()[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break'''

    print('PROCESSED IMAGES', idx)

    if save:
        filepath = working_dir.strip('detectron2') + 'dataset/unripe.json'
        with open(filepath, 'w') as f:
            json.dump(unann, f)

    return unann


test_detectron2(img_dir, save=True)
