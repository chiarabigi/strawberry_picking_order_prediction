import numpy as np
import torch
from utils.utils import get_single_out
import matplotlib.pyplot as plt
import os
import copy
from collections import Counter

base_path = os.path.dirname(os.path.abspath(__file__))


def get_comparison(pred, batch, sched_easiness, sched_students, sched_heuristic, occ_1):
    sx = 0
    batch_size = int(batch.batch[-1]) + 1

    guessed_easiness = 0
    guessed_heuristic = 0
    guessed_students = 0
    sched_guessed_easiness = [0] * 33
    sched_guessed_heuristic = [0] * 33
    sched_guessed_students = [0] * 33
    for j in range(batch_size):
        dx = get_single_out(batch.batch, j, sx)
        y_occ = batch.info[sx:dx].cpu().detach().numpy().transpose()[0]
        y_ea = batch.easiness_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_heu = batch.heuristic_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_stud = batch.students_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_pred = pred[sx:dx].cpu().detach().numpy().transpose()[0]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])
        sched = [x + 1 for x in sched]

        unripe = np.unique(y_stud).size

        for k in range(len(sched)):
            sched_easiness[y_ea[k] - 1][sched[k] - 1] += 1
            sched_heuristic[y_heu[k] - 1][sched[k] - 1] += 1
            occ_1[y_occ[k]][sched[k] - 1] += 1
            if sched[k] == y_ea[k]:
                sched_guessed_easiness[sched[k] - 1] += 1
            if sched[k] == y_heu[k]:
                sched_guessed_heuristic[sched[k] - 1] += 1
            if y_stud[k] == unripe and sched[k] > unripe:
                sched_value = y_stud[k]
            else:
                sched_value = sched[k]
            sched_students[y_stud[k] - 1][sched_value - 1] += 1
            if sched_value == y_stud[k]:
                sched_guessed_students[sched_value - 1] += 1

        guessed_easiness += 100 * sum([1 for i in range(len(sched)) if (sched[i] == y_ea[i] and y_ea[i] <= unripe)]) / unripe
        guessed_heuristic += 100 * sum([1 for i in range(len(sched)) if (sched[i] == y_heu[i] and y_heu[i] <= unripe)]) / unripe
        guessed_students += 100 * sum([1 for i in range(len(sched)) if (sched[i] == y_stud[i] and y_stud[i] <= unripe)]) / unripe
        sx = dx

    # weighted per len of graph
    tot_guessed_easiness = guessed_easiness / (batch_size - 1)
    tot_guessed_heuristic = guessed_heuristic / (batch_size - 1)
    tot_guessed_students = guessed_students / (batch_size - 1)

    listheuristic = sorted(
        Counter([item for sublist in batch.heuristic_ann.tolist() for item in sublist]).most_common())
    listeasiness = sorted(
        Counter([item for sublist in batch.easiness_ann.tolist() for item in sublist]).most_common())
    liststudents = sorted(
        Counter([item for sublist in batch.students_ann.tolist() for item in sublist]).most_common())
    how_guessed_heuristic = [100 * sched_guessed_heuristic[i] / listheuristic[i][1] for i in range(len(listheuristic))]
    how_guessed_easiness = [100 * sched_guessed_easiness[i] / listeasiness[i][1] for i in range(len(listeasiness))]
    how_guessed_students = [100 * sched_guessed_students[i] / liststudents[i][1] for i in range(len(liststudents))]

    percentage_matrix_heu = copy.copy(sched_heuristic)
    for i in range(len(sched_heuristic)):
        for j in range(len(sched_heuristic[i])):
            if sum(sched_heuristic[:, j]) == 0:
                value = 0
            else:
                value = int(100 * sched_heuristic[i, j] / sum(sched_heuristic[:, j]))
            percentage_matrix_heu[i, j] = value
    correctens_heu = sum(percentage_matrix_heu[i, i] for i in range(len(percentage_matrix_heu))) / sum(
        1 for i in range(len(percentage_matrix_heu)) if
        (sum(percentage_matrix_heu[i]) != 0 and sum(percentage_matrix_heu[:, i]) != 0))

    percentage_matrix_easy = copy.copy(sched_easiness)
    for i in range(len(sched_easiness)):
        for j in range(len(sched_easiness[i])):
            if sum(sched_easiness[:, j]) == 0:
                value = 0
            else:
                value = int(100 * sched_easiness[i, j] / sum(sched_easiness[:, j]))
            percentage_matrix_easy[i, j] = value
    correctens_easy = sum(percentage_matrix_easy[i, i] for i in range(len(percentage_matrix_easy))) / sum(
        1 for i in range(len(percentage_matrix_easy)) if
        (sum(percentage_matrix_easy[i]) != 0 and sum(percentage_matrix_easy[:, i]) != 0))

    percentage_matrix_stud = copy.copy(sched_students)
    for i in range(len(sched_students)):
        for j in range(len(sched_students[i])):
            if sum(sched_students[:, j]) == 0:
                value = 0
            else:
                value = int(100 * sched_students[i, j] / sum(sched_students[:, j]))
            percentage_matrix_stud[i, j] = value
    correctens_stud = sum(percentage_matrix_stud[i, i] for i in range(len(percentage_matrix_stud))) / sum(
        1 for i in range(len(percentage_matrix_stud)) if
        (sum(percentage_matrix_stud[i]) != 0 and sum(percentage_matrix_stud[:, i]) != 0))

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
        diag = sum(percentage_matrix[i, i] for i in range(len(percentage_matrix)))\
              / sum(1 for i in range(len(percentage_matrix))
                    if (sum(percentage_matrix[i]) != 0 and sum(percentage_matrix[:, i]) != 0))
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
