import numpy as np
import torch
from utils.utils import get_single_out
import matplotlib.pyplot as plt
import os
import copy

base_path = os.path.dirname(os.path.abspath(__file__))


def get_comparison(pred, batch, sched_easiness, sched_students, sched_heuristic, occ_1):
    sx = 0
    batch_size = int(batch.batch[-1]) + 1

    for j in range(batch_size):
        dx = get_single_out(batch.batch, j, sx)
        y_occ = batch.info[sx:dx].cpu().detach().numpy().transpose()[0]
        y_ea = batch.easiness_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_heu = batch.heuristic_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_stud = batch.students_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_pred = pred[sx:dx].cpu().detach().numpy().transpose()[0]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])

        unripe = np.unique(y_stud).size + 1

        for k in range(len(sched)):
            sched_easiness[y_ea[k] - 1][sched[k]] += 1
            sched_heuristic[y_heu[k] - 1][sched[k]] += 1
            occ_1[y_occ[k]][sched[k]] += 1
            if y_stud[k] == len(y_stud) - unripe + 1 and sched[k] > (len(sched) - unripe):
                sched[k] = y_stud[k] - 1
            sched_students[y_stud[k] - 1][sched[k]] += 1
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
    plt.xlabel('Predicted scheduling')
    plt.ylabel(name)
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

    percentage_matrix = copy.copy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if sum(matrix[:, j]) == 0:
                value = 0
            else:
                value = int(100 * matrix[i, j] / sum(matrix[:, j]))
            percentage_matrix[i, j] = value
            text = ax.text(j, i, value,
                           ha="center", va="center", color="w", fontsize='xx-small')

    if len(matrix[0]) == len(matrix[:, 0]):
        diag = sum(percentage_matrix[i, i] for i in range(len(matrix))) / len(matrix)
        plt.suptitle('% of predicted scheduling is actually: ' + name)
        plt.title('% of correspondence: {}'.format(diag))
        ith_correspondence = [100 * abs(sum(matrix[i]) - abs(sum(matrix[i]) - sum(matrix[:, i]))) / sum(matrix[i]) for i in range(len(matrix))]
        print(name)
        print(ith_correspondence)
    else:
        plt.title('% of occlusion property for each scheduling prediction')

    # Displaying the plot
    plt.savefig(base_path.strip('utils') + '/imgs/heatmaps/heatmap{}.png'.format(name))
    plt.close()
