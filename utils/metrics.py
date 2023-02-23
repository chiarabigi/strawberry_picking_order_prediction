import numpy as np
import torch
from utils import get_single_out


def get_occlusion1(output, occ, batch):
    sx = 0
    batch_size = int(batch[-1]) + 1
    occlusion = np.zeros(5)
    for i in range(batch_size):
        dx = get_single_out(batch, i, sx)
        y_pred = output[sx:dx]
        y_occ = occ[sx:dx]
        index = y_pred.argmax(0)
        occlusion[y_occ[index]] += 1
        sx = dx
    return occlusion

def get_realscheduling(output, label, batch):
    sx = 0
    batch_size = int(batch[-1]) + 1
    scheduling = np.zeros(18)
    for i in range(batch_size):
        dx = get_single_out(batch, i, sx)
        y_pred = output[sx:dx]
        y_lab = label[sx:dx]
        index = y_pred.argmax(0)
        scheduling[y_lab[index] - 1] += 1
        sx = dx
    return scheduling

def get_whole_scheduling(output, label, batch):
    sx = 0
    batch_size = int(batch[-1]) + 1
    sched_pred = np.zeros(18)
    sched_true = np.zeros(18)
    for i in range(batch_size):
        dx = get_single_out(batch, i, sx)
        y_pred = output[sx:dx]
        y_lab = label[sx:dx]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])
        for j in range(len(sched)):
            sched_pred[sched[j] - 1] += 1
        for k in range(len(y_lab)):
            sched_true[y_lab[k] - 1] += 1
        sx = dx
    return sched_pred, sched_true

def get_label_scheduling(label, batch):
    batch_size = int(batch[-1]) + 1
    sched_true = np.zeros(18)
    for i in range(batch_size):
        y_lab = label[i]
        for k in range(len(y_lab)):
            sched_true[y_lab[k] - 1] += 1
    return sched_true


class BatchAccuracy_scheduling(torch.nn.Module):
    def __init__(self):
        super(BatchAccuracy_scheduling, self).__init__()

    def forward(self, all_y_pred, all_y_true, batch):
        sx = 0
        batch_size = int(batch[-1]) + 1
        corrects = 0
        for i in range(batch_size):
            dx = get_single_out(batch, i, sx)
            y_pred = all_y_pred[sx:dx]
            y_true = all_y_true[sx:dx]
            for j in range(len(y_true)):
                if y_pred[j] > 0.5:
                    if y_true[j] >= 0.9:
                        corrects += 1
                else:
                    if y_true[j] < 0.5:
                        corrects += 1
            sx = dx
        return corrects
