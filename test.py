'''
Test a model! Chose one from the 'best_models' folder, and copy at the end of the 'model_path' variable.
Go to config.py to choose correctly the name of the approach used to train the model.
'''

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import os
import matplotlib.pyplot as plt
from collections import Counter
import config as cfg
from utils.metrics import get_comparison, plot_heatmap

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))

test_path = base_path + '/dataset/data_test/'
test_dataset = cfg.DATASET(test_path)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = cfg.MODEL(cfg.HL, cfg.NL).to(device)
model_path = base_path + '/best_models/model_20230224_101010.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

sched_easiness = np.zeros((32, 32))
sched_students = np.zeros((32, 32))
sched_heuristic = np.zeros((32, 32))
occ_1 = np.zeros((3, 32))

gt = []
y = []
for i, tbatch in enumerate(test_loader):
    pred = model(tbatch)
    if cfg.approach != 'class':
        gt += tbatch.y.tolist()
        y += pred.tolist()
    else:
        sched_easiness, sched_students, sched_heuristic, occ_1 = get_comparison(pred, tbatch, sched_easiness,
                                                                                sched_students, sched_heuristic, occ_1)

if cfg.approach != 'class':
    # Print matches
    wT = Counter([item for sublist in gt for item in sublist])
    value0T = list(wT.values())[list(wT.keys()).index(0)]
    wTk = wT.keys()
    wTv = wT.values()
    list(wTk).remove(0)
    list(wTv).remove(value0T)
    plt.bar(wTk, wTv, width=0.001)
    wP = Counter([item for sublist in y for item in sublist])
    wPk = wP.keys()
    wPv = wP.values()
    try:
        value0P = list(wP.values())[list(wP.keys()).index(0)]
        list(wPk).remove(0)
        list(wPv).remove(value0P)
    except ValueError:
        a = 'a'
    plt.bar(wPk, wPv, width=0.001)
    plt.title('Strawberry test easiness score. Blue: gt. Orange: predicted')
    plt.savefig('imgs/truetestEasiness.png')
else:
    plot_heatmap(sched_easiness, list(range(0, 31)), 'heuristic easiness score scheduling')
    plot_heatmap(sched_heuristic, list(range(0, 31)), 'heuristic min max scheduling')
    plot_heatmap(sched_students, list(range(0, 31)), 'students scheduling')
    plot_heatmap(occ_1, ['NON', ' BY LEAF', 'BY BERRY'], 'occlusion property')
