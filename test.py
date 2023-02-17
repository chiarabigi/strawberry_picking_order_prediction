import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config_scheduling as cfg
import config_scheduling_gpu
from utils import get_occlusion1, get_realscheduling, get_whole_scheduling, get_label_scheduling
import multiprocessing
from model import GCN_scheduling
import json
from utils import get_single_out

def get_all_realscheduling(output, label, batch):
    sx = 0
    batch_size = int(batch[-1]) + 1
    scheduling = np.zeros((18, 18))
    for i in range(batch_size):
        dx = get_single_out(batch, i, sx)
        y_pred = output[sx:dx]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])
        y_lab = label[sx:dx]
        for j in range(len(sched)):
            index = sched[j]
            scheduling[index][y_lab[index] - 1] += 1
        sx = dx
    return scheduling

gnn_path = '/home/chiara/strawberry_picking_order_prediction/dataset/easiness/data_test/raw/gnnann.json'
with open(gnn_path) as f:
    anns = json.load(f)
infoT = {k: [dic[k] for dic in anns] for k in anns[0]}
students_ann = infoT['students_sc_ann']
heuristic_ann = infoT['heuristic_sc_ann']

test_path = 'dataset/easiness/data_test/'
test_dataset = cfg.DATASET(test_path)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                         shuffle=False)  # , num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
model = GCN_scheduling(8, 0)

criterion = ['bce', 'mse']
preds = []
batches = []
for c in criterion:
    print(c)
    model_path = '/home/chiara/strawberry_picking_order_prediction/best_models/model_easiness_{}.pth'.format(c)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    real_tscheduling = np.zeros(18)
    all_real_tscheduling = np.zeros((18, 18))
    sched_pred = np.zeros(18)
    sched_true = np.zeros(18)
    sched_students = np.zeros(18)
    sched_heuristic = np.zeros(18)
    occ_1 = np.zeros(5)

    for i, tbatch in enumerate(test_loader):
        # tbatch.to(device)
        pred = model(tbatch)
        real_tscheduling += get_realscheduling(pred, tbatch.label, tbatch.batch)
        all_real_tscheduling += get_all_realscheduling(pred, tbatch.label, tbatch.batch)
        s_pred, s_true = get_whole_scheduling(pred, tbatch.label, tbatch.batch)
        sched_students += get_label_scheduling(students_ann, tbatch.batch)
        sched_heuristic += get_label_scheduling(heuristic_ann, tbatch.batch)
        sched_pred += s_pred
        sched_true += s_true
        occ_1 += get_occlusion1(pred, tbatch.info, tbatch.batch)
        preds.append(pred)
        batches.append(tbatch.batch)


    for i in range(len(real_tscheduling) - 1):
        print(f'\n {100*real_tscheduling[i]/real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where scheduled as {i + 1:.4f}')
    print(f'\n {100*real_tscheduling[-1]/real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where UNRIPE')
    '''
    occlusion_properties = ['NON OCCLUDED', 'OCCLUDED BY LEAF', 'OCCLUDED BY A BERRY']
    occ = np.zeros(3)
    occ[0] += occ_1[1] + occ_1[3]
    occ[1] += occ_1[0] + occ_1[2]
    occ[2] += occ_1[4]
    for i in range(len(occ)):
        print(f'\n {100*occ[i]/occ.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where labeled as {occlusion_properties[i]}')
    
    sched_pred += 2
    sched_students += 2
    sched_true += 2
    sched_heuristic += 2
    print('\nPREDICTION VS STUDENTS')
    for i in range(len(sched_pred) - 1):
        print(f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_students[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as students annotation')
    print(f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_students[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as students annotation')
    print('\nHEURISTIC')
    for i in range(len(sched_pred) - 1):
        print(f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_heuristic[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as heuristic')
    print(f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_heuristic[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as heuristic')
    print('\nPREDICTION VS GROUND TRUTH  (=scheduling from easiness: first is easiest, last is least easy...)')
    for i in range(len(sched_pred) - 1):
        print(f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_true[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as ground truth')
    print(f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_true[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as ground truth')
    '''
one = 1