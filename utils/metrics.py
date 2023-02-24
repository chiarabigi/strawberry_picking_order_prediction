import numpy as np
import torch
from utils.utils import get_single_out
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(os.path.abspath(__file__))

def get_comparison(pred, batch, sched_easiness, sched_students, sched_heuristic, occ_1):
    sx = 0
    batch_size = int(batch.batch[-1]) + 1

    for j in range(batch_size):
        dx = get_single_out(batch.batch, j, sx)
        y_occ = batch.info[sx:dx]
        y_ea = batch.easiness_ann[sx:dx]
        y_heu = batch.heuristic_ann[sx:dx]
        y_stud = batch.students_ann[sx:dx]
        y_pred = pred[sx:dx]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])

        unripe = len(y_stud) - len(list(set(y_stud))) + 1

        for k in range(len(sched)):
            sched_easiness[sched[k]][y_ea[k] - 1] += 1
            occ_1[y_occ[k]][sched[k]] += 1
            if y_stud[k] == len(y_stud) - unripe + 1 and sched[k] > (len(sched) - unripe):
                sched[k] = y_stud[k] - 1
            sched_students[sched[k]][y_stud[k] - 1] += 1
            sched_heuristic[sched[k]][y_heu[k] - 1] += 1
        sx = dx

    return sched_easiness, sched_students, sched_heuristic, occ_1


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

def plot_heatmap(matrix, y_ticks, name):
    # Generating data for the heat map STUDENTS
    data = matrix
    fig, ax = plt.subplots()
    plt.imshow(data)
    # Adding details to the plot
    plt.title('Predicted scheduling VS ' + name)
    plt.xlabel('Predicted scheduling')
    plt.ylabel(name)
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            text = ax.text(j, i, int(matrix[i, j]),
                           ha="center", va="center", color="w", fontsize='x-small')

    # Displaying the plot
    plt.savefig(base_path.strip('utils') + '/imgs/heatmaps/heatmap{}.png'.format(name))
    plt.close()
