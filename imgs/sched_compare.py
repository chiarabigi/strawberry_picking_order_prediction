import numpy as np
import json
import os
import matplotlib.pyplot as plt
import copy

base_path = os.path.dirname(os.path.abspath(__file__))

students_ann = []
heuristic_ann = []
easiness_ann = []
heuristic_ann_ripe = []
easiness_ann_ripe = []
length = []
phases = ['train', 'test', 'val']
for phase in phases:
    gnn_path = base_path.strip('imgs') + '/dataset/data_{}/raw/gnnann.json'.format(phase)
    with open(gnn_path) as f:
        anns = json.load(f)
    infoT = {k: [dic[k] for dic in anns] for k in anns[0]}
    students_ann += infoT['students_sc_ann']
    heuristic_ann += infoT['heuristic_sc_ann']
    length.append(len(heuristic_ann))
    easiness_ann += infoT['easiness_sc_ann']
    heuristic_ann_ripe = copy.copy(heuristic_ann)
    easiness_ann_ripe = copy.copy(easiness_ann)
    for e in range(len(easiness_ann)):
        unripe = np.unique(students_ann[e]).size
        if len(students_ann[e]) == unripe:
            unripe += 1
        students_ann[e] = [x for x in students_ann[e] if x < unripe]
        heuristic_ann_ripe[e] = [x for x in heuristic_ann[e] if x < unripe]
        easiness_ann_ripe[e] = [x for x in easiness_ann[e] if x < unripe]

sched_stu_ea = np.zeros((17, 17))
sched_heu_ea = np.zeros((33, 33))
heuristic_heu_stu = np.zeros((17, 17))
for l in range(len(easiness_ann)):
    y_ea = easiness_ann[l]
    y_heu = heuristic_ann[l]
    y_ea_r = easiness_ann_ripe[l]
    y_heu_r = heuristic_ann_ripe[l]
    y_stud = students_ann[l]

    for k in range(len(y_ea)):
        sched_heu_ea[y_ea[k] - 1][y_heu[k] - 1] += 1
        if k < len(y_ea_r) - 1:
            sched_stu_ea[y_ea_r[k] - 1][y_stud[k] - 1] += 1
            heuristic_heu_stu[y_heu_r[k] - 1][y_stud[k] - 1] += 1

# Generating data for the heat map HEURISTIC
data = sched_heu_ea
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("% of easiness is actually heuristic scheduling")
plt.ylabel('heuristic easiness score')
plt.xlabel('heuristic min max')
plt.colorbar()
for i in range(len(sched_heu_ea)):
    for j in range(len(sched_heu_ea)):
        if sum(sched_heu_ea[:, j]) == 0:
            value = 0
        else:
            value = int(100 * sched_heu_ea[i, j] / sum(sched_heu_ea[:, j]))
        text = ax.text(j, i, value, ha="center", va="center", color="w", fontsize='xx-small')

# Displaying the plot
plt.savefig(base_path + '/heatmaps/different_schedulings_comparison/compareHeuristicEasiness.png')

# Generating data for the heat map STUDENTS
data = sched_stu_ea
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("% of students is actually easiness scheduling")
plt.ylabel('heuristic easiness score')
plt.xlabel('students scheduling')
plt.colorbar()
for i in range(len(sched_stu_ea)):
    for j in range(len(sched_stu_ea)):
        if sum(sched_stu_ea[:, j]) == 0:
            value = 0
        else:
            value = int(100 * sched_stu_ea[i, j] / sum(sched_stu_ea[:, j]))
        text = ax.text(j, i, value, ha="center", va="center", color="w", fontsize='small')

# Displaying the plot
plt.savefig(base_path + '/heatmaps/different_schedulings_comparison/compareStudentsEasiness.png')

# Generating data for the heat map STUDENTS
data = heuristic_heu_stu
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("% of students is actually heuristic scheduling")
plt.ylabel('heuristic min max')
plt.xlabel('students scheduling')
plt.colorbar()
for i in range(len(heuristic_heu_stu)):
    for j in range(len(heuristic_heu_stu)):
        if sum(heuristic_heu_stu[:, j]) == 0:
            value = 0
        else:
            value = int(100 * heuristic_heu_stu[i, j] / sum(heuristic_heu_stu[:, j]))
        text = ax.text(j, i, value, ha="center", va="center", color="w", fontsize='small')

# Displaying the plot
plt.savefig(base_path + '/heatmaps/different_schedulings_comparison/compareStudentsHeuristic.png')
