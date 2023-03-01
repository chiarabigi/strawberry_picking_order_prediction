'''
INPUT: raw image
OUTPUT: first strawberry to be picked, that will be succesfully picked

HOW?
1) image to DETR --> strawberry identification and classification
2) strawberries in graph1 representation
3) graph1 to scheduling prediction GAT model --> which is the target strawberry
4) strawberries + target info in graph2 representation
5) graph2 to success of picking GAT model --> will the target strawberry be picked
    If negative output:
4)  strawberries + second target (= second highest probability of being first) info in graph2 representation
5) ...
    If positive output never arrives: ask for another raw image!
'''

import torch
from dataset import PickingSuccessDataset
import json
import os
import config as cfg_s
import config_sop as cfg_p
import time
from detr.test import test_detr
from detr_to_gnnann_sop import ann_to_gnnann
# from detectron2.inference_dyson_keypoints import test_detectron2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.path.dirname(os.path.abspath(__file__))


def unite_infos(json_annotations_path, target):
    with open(json_annotations_path+'/raw/gnnann.json') as f:
        anns = json.load(f)
    anns = anns[0]
    target_vector = [False] * len(anns[0])
    target_vector[target] = True
    anns['target_vector'] = target_vector
    json_annotations_path = base_path + '/experiment_test/raw/gnnann.json'
    with open(json_annotations_path, "w") as f:
        json_str = json.dumps(anns)
        f.write(json_str)

    return json_annotations_path


class Experiment(torch.nn.Module):
    def __init__(self):
        super(Experiment, self).__init__()

        self.best_GAT_scheduling_model = cfg_s.MODEL(cfg_s.HL, cfg_s.NL)
        self.best_GAT_scheduling_model.load_state_dict(
            torch.load(base_path.strip('success_of_picking') + '/best_models/model_20230224_132115.pth'))

        self.best_GAT_picking_success_model = cfg_p.MODEL
        self.best_GAT_picking_success_model.load_state_dict(
            torch.load(base_path.strip('success_of_picking') + '/best_models/model_success_of_picking.pth'))

    def forward(self, image_path):

        # obtain bounding boxes of unripe strawberries from raw image
        unripe_info = [{
            'file_name': [],
            'bboxes': []
        }]  # test_detectron2(image_path, save=False)

        # obtain bounding boxes and occlusion properties from raw image
        occlusion_info = test_detr(image_path)

        json_annotations_path = ann_to_gnnann(occlusion_info, unripe_info, image_path)

        graph_data_scheduling = cfg_s.DATASET(json_annotations_path)

        start_time = time.time()  # set the time at which inference started

        scheduling_probability_vector = self.best_GAT_scheduling_model(graph_data_scheduling.get(0)).squeeze(1)

        stop_time = time.time()
        duration = stop_time - start_time
        hours = duration // 3600
        minutes = (duration - (hours * 3600)) // 60
        seconds = duration - ((hours * 3600) + (minutes * 60))
        msg = (f'SCHEDULING training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {float(seconds):4.2f} seconds')
        print(msg, flush=True)  # print out inferenceduration time

        success_of_picking = 0
        attempt = 0

        while not(success_of_picking) or attempt < len(scheduling_probability_vector):

            json_annotations2_path = unite_infos(json_annotations_path, scheduling_probability_vector.argmax(0))  # adds target information

            graph_data_picking_success = PickingSuccessDataset('/'+json_annotations2_path.strip('/raw/gnnann.json'))

            start_time = time.time()  # set the time at which inference started
            prob_success_of_picking = self.best_GAT_picking_success_model(graph_data_picking_success.get(0))
            stop_time = time.time()
            duration = stop_time - start_time
            hours = duration // 3600
            minutes = (duration - (hours * 3600)) // 60
            seconds = duration - ((hours * 3600) + (minutes * 60))
            msg = (f'PICKING SUCCESS training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {float(seconds):4.2f} seconds')
            print(msg, flush=True)  # print out inferenceduration time

            attempt += 1

            if prob_success_of_picking > 0.5:
                success_of_picking = 1
            else:
                scheduling_probability_vector[scheduling_probability_vector.argmax()] = 0


        if success_of_picking:
            target_strawberry = scheduling_probability_vector.argmax(1)
        else:
            print('I cannot pick anything in this cluster, feed me another raw image and try again!')
            target_strawberry = -1

        return target_strawberry  # I want maybe an image with a circle on the target?


experiment = Experiment()
image_path = ''
target_strawberry = experiment(image_path)

