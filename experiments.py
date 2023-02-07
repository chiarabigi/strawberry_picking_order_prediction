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
from torch_geometric.data import DataLoader
from dataset import SchedulingDataset, PickingSuccessDataset
from utils import unite_infos
import config_scheduling as cfg_s
import config_picking_success as cfg_p
import time


class Experiment(torch.nn.Module):
    def __init__(self):
        super(Experiment, self).__init__()
        # self.best_detr_model = torch.load('path/to/best_detr_model.pt')

        self.best_GAT_scheduling_model = cfg_s.MODEL
        self.best_GAT_scheduling_model.load_state_dict(torch.load('/home/chiara/SCHEDULING/GNN/best_models/best_models_scheduling/model_20230131_171843'))

        self.best_GAT_picking_success_model = cfg_p.MODEL
        self.best_GAT_picking_success_model.load_state_dict(torch.load('/home/chiara/SCHEDULING/GNN/best_models/best_models_robofruit/model_20230131_160549'))

    def forward(self, raw_image):

        # json_annotations_path = self.best_detr_model(raw_image)  # I want directly gnnann!
        json_annotations_path = '/home/chiara/SCHEDULING/GNN/dataset/scheduling/data_test'

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


raw_image = 'raw/image/path/.png'
experiment = Experiment()
target_strawberry = experiment(raw_image)

