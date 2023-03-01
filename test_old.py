'''
Test a model! Chose one from the 'best_models' folder, and copy at the end of the 'model_path' variable.
Go to config.py to choose correctly the name of the approach used to train the model.
'''

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import json
import os
from model import GCN_OLD_scheduling
from utils.metrics import get_comparison, plot_heatmap
from data_scripts.old_dataset import SchedulingDataset
from utils.old_metrics import get_realscheduling, get_label_scheduling, get_whole_scheduling, get_comparison

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))

with open(base_path + '/dataset/OLD/target/data_test/raw/gnnann.json') as f:
    annT = json.load(f)
ann = {k: [dic[k] for dic in annT] for k in annT[0]}

test_path = base_path + '/dataset/OLD/target/data_test'
test_dataset = SchedulingDataset(test_path)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = GCN_OLD_scheduling(8, 0).to(device)
model_path = base_path + '/best_models/model_best_sched.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

sched_students = np.zeros((32, 32))
for i, tbatch in enumerate(test_loader):
    pred = model(tbatch)

    real_scheduling = get_realscheduling(pred, tbatch.label, tbatch.batch)
    sched_pred, sched_true = get_whole_scheduling(pred, tbatch.label, tbatch.batch)
    sched_true2 = get_label_scheduling(tbatch.label, tbatch.batch)
    sched_students = get_comparison(pred, tbatch.batch, tbatch.label, sched_students)

sched_pred = [x + 2 for x in sched_pred]
sched_true = [x + 2 for x in sched_true]
correspondance = [100 * abs(sched_pred[i] - (abs(sched_pred[i] - sched_true[i]))) / sched_pred[i] for i in range(len(sched_pred))]
one = 1
#plot_heatmap(sched_easiness, list(range(0, 31)), 'heuristic easiness score scheduling')
#plot_heatmap(sched_heuristic, list(range(0, 31)), 'heuristic min max scheduling')
plot_heatmap(sched_students, list(range(0, 31)), 'students scheduling OLD MODEL')
#plot_heatmap(occ_1, ['NON', ' BY LEAF', 'BY BERRY'], 'occlusion property')
