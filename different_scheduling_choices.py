import torch
import json
from torch_geometric.data import DataLoader
import config_picking_success as cfg_p
from scipy.spatial import distance
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy = 'rightest'

def leftest(boxes):
    minL = 1_000_000.
    target = 0
    for i in range(len(boxes)):
        L = distance.euclidean((0, 0), (boxes[i][0], boxes[i][1]))
        if L < minL:
            minL = L
            target = i
    return target

def rightest(boxes):
    maxL = 0.0
    target = 0
    for i in range(len(boxes)):
        L = distance.euclidean((0, 0), (boxes[i][0], boxes[i][1]))
        if L > maxL:
            maxL = L
            target = i
    return target


def unite_infos(json_annotations_path):
    with open(json_annotations_path+'/raw/gnnann.json') as f:
        annotations = json.load(f)
    new_anns = []
    for i in range(len(annotations)):
        anns= annotations[i]
        target_vector = [False] * len(anns[0])
        if policy == 'leftest':
            target = leftest(anns[0])
        elif policy == 'annotated':
            target = anns[1].index(min(anns[1]))
        elif policy == 'rightest':
            target = rightest(anns[0])
        elif policy == 'random':
            target = random.randrange(0, len(anns[0]))
        target_vector[target] = True
        new_anns.append([anns[0], anns[2], target_vector, 0.0])
    json_annotations_path = '/home/chiara/SCHEDULING/GNN/choices_test/{}/raw/gnnann.json'.format(policy)
    with open(json_annotations_path, "w") as f:
        json_str = json.dumps(new_anns)
        f.write(json_str)
    return json_annotations_path


class Choices(torch.nn.Module):
    def __init__(self):
        super(Choices, self).__init__()

        self.best_GAT_picking_success_model = cfg_p.MODEL
        self.best_GAT_picking_success_model.load_state_dict(torch.load('/home/chiara/SCHEDULING/GNN/best_models/best_models_robofruit/model_20230131_160549'))

    def forward(self):

        json_annotations_path = '/home/chiara/SCHEDULING/GNN/dataset/scheduling/data_test'
        json_annotations2_path = unite_infos(json_annotations_path)  # adds target information
        graph_data_picking_success = cfg_p.DATASET('/home/chiara/SCHEDULING/GNN/choices_test/{}'.format(policy))

        # Take target as first:
        # graph_data_picking_success = cfg_p.DATASET('/home/chiara/SCHEDULING/GNN/dataset/picking_success/data_test')

        graph_data = DataLoader(graph_data_picking_success, batch_size=len(graph_data_picking_success))

        success_of_picking = 0
        targets = 0
        for i, batch in enumerate(graph_data):
            prob_success_of_picking = self.best_GAT_picking_success_model(batch.to(device))
            for i in range(len(prob_success_of_picking)):
                if prob_success_of_picking[i] > 0.5:
                    success_of_picking += 1
            targets += len(prob_success_of_picking)

        print(success_of_picking)
        print(targets)
        success_rate = float(success_of_picking/targets)
        print(float(success_of_picking/targets))


choices = Choices()
choices()

