import numpy as np
import json
import matplotlib.pyplot as plt


students_ann = []
heuristic_ann = []
easiness_ann = []
length = []
phases = ['train', 'test', 'val']
for phase in phases:
    gnn_path = '/home/chiara/strawberry_picking_order_prediction/dataset/data_{}/raw/gnnann.json'.format(phase)
    with open(gnn_path) as f:
        anns = json.load(f)
    infoT = {k: [dic[k] for dic in anns] for k in anns[0]}
    students_ann += infoT['students_sc_ann']
    heuristic_ann += infoT['heuristic_sc_ann']
    length.append(len(heuristic_ann))
    easiness_ann += infoT['sc_ann']
    for e in range(len(easiness_ann)):
        unripe = [x for x in heuristic_ann[e] if x == 18]
        students_ann[e] = [x for x in students_ann[e] if x != 18]
        heuristic_ann[e] = [x for x in heuristic_ann[e] if x != 18]
        easiness_ann[e] = [x for x in easiness_ann[e] if x < len(easiness_ann[e]) - len(unripe)]

sched_students = np.zeros((17, 17))
sched_heuristic = np.zeros((17, 17))
heuristic_students = np.zeros((17, 17))
for l in range(len(easiness_ann)):
    sched = easiness_ann[l]
    y_heu = heuristic_ann[l]
    y_stud = students_ann[l]

    for k in range(len(sched)):
        sched_students[sched[k] - 1][y_stud[k] - 1] += 1
        sched_heuristic[sched[k] - 1][y_heu[k] - 1] += 1
        heuristic_students[y_heu[k] - 1][y_stud[k] - 1] += 1

# Generating data for the heat map HEURISTIC
data = sched_heuristic
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Comparison of scheduled strawberries")
plt.ylabel('easiness_scheduling')
plt.xlabel('heuristic_scheduling')
plt.colorbar()
for i in range(len(sched_heuristic)):
    for j in range(len(sched_heuristic)):
        text = ax.text(j, i, int(sched_heuristic[i, j]),
                       ha="center", va="center", color="w", fontsize='small')

# Displaying the plot
plt.savefig('imgs/heatmaps/compareHeuristicEasiness.png')

# Generating data for the heat map STUDENTS
data = sched_students
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Comparison of scheduled strawberries")
plt.ylabel('easiness_scheduling')
plt.xlabel('students_scheduling')
plt.colorbar()
for i in range(len(sched_students)):
    for j in range(len(sched_students)):
        text = ax.text(j, i, int(sched_students[i, j]),
                       ha="center", va="center", color="w", fontsize='small')

# Displaying the plot
plt.savefig('imgs/heatmaps/compareStudentsEasiness.png')

# Generating data for the heat map STUDENTS
data = heuristic_students
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Comparison of scheduled strawberries")
plt.ylabel('heuristic_scheduling')
plt.xlabel('students_scheduling')
plt.colorbar()
for i in range(len(heuristic_students)):
    for j in range(len(heuristic_students)):
        text = ax.text(j, i, int(heuristic_students[i, j]),
                       ha="center", va="center", color="w", fontsize='small')

# Displaying the plot
plt.savefig('imgs/heatmaps/compareStudentsHeuristic.png')
