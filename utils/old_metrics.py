import numpy as np
from utils.utils import get_single_out
from collections import Counter
import copy


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
            sched_pred[sched[j]] += 1
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


def get_comparison(pred, batch_batch, batch_students_ann, sched_students):
    sx = 0
    batch_size = int(batch_batch[-1]) + 1

    guessed_students = 0
    sched_guessed_students = [0] * 33
    for j in range(batch_size):
        dx = get_single_out(batch_batch, j, sx)
        y_stud = batch_students_ann[sx:dx].cpu().detach().numpy().transpose()[0]
        y_pred = pred[sx:dx].cpu().detach().numpy().transpose()[0]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])
        sched = [x + 1 for x in sched]

        unripe = np.unique(y_stud).size
        if len(y_stud) == unripe:
            unripe += 1

        for k in range(len(sched)):
            if y_stud[k] == unripe and sched[k] > unripe:
                sched_value = y_stud[k]
            else:
                sched_value = sched[k]
            sched_students[y_stud[k] - 1][sched_value - 1] += 1
            if sched_value == y_stud[k]:
                sched_guessed_students[sched_value - 1] += 1

        guessed_students += 100 * sum([1 for i in range(len(sched)) if (sched[i] == y_stud[i] and y_stud[i] <= unripe)]) / unripe
        sx = dx

    # weighted per len of graph
    tot_guessed_students = guessed_students / (batch_size - 1)

    liststudents = sorted(
        Counter([item for sublist in batch_students_ann.tolist() for item in sublist]).most_common())
    how_guessed_students = [100 * sched_guessed_students[i] / liststudents[i][1] for i in range(len(liststudents))]

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

    return sched_students
