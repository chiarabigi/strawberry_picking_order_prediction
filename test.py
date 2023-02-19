import numpy as np
import torch
from torch_geometric.loader import DataLoader
import config_scheduling as cfg
from model import GCN_scheduling
import json
from utils import get_single_out
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

gnn_path = '/home/chiara/strawberry_picking_order_prediction/dataset/data_test/raw/gnnann.json'
with open(gnn_path) as f:
    anns = json.load(f)
infoT = {k: [dic[k] for dic in anns] for k in anns[0]}
students_ann = infoT['students_sc_ann']
heuristic_ann = infoT['heuristic_sc_ann']

test_path = 'dataset/data_test/'
test_dataset = cfg.DATASET(test_path)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                         shuffle=False)  # , num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
model = GCN_scheduling(8, 0)


model_path = '/home/chiara/strawberry_picking_order_prediction/best_models/model_20230218_122931.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
all_real_tscheduling = np.zeros((18, 18))
sched_true = np.zeros((18, 18))
sched_students = np.zeros((18, 18))
sched_heuristic = np.zeros((18, 18))
occ_1 = np.zeros((4, 18))

for i, tbatch in enumerate(test_loader):
    # tbatch.to(device)
    pred = model(tbatch)
    sx = 0
    batch_size = int(tbatch.batch[-1]) + 1

    for j in range(batch_size):
        dx = get_single_out(tbatch.batch, j, sx)
        y_occ = tbatch.info[sx:dx]
        y_gt = tbatch.label[sx:dx]
        y_heu = tbatch.heuristic_ann[sx:dx]
        y_stud = tbatch.students_ann[sx:dx]
        y_pred = pred[sx:dx]
        sched = sorted(range(len(y_pred)), reverse=True, key=lambda k: y_pred[k])

        unripe = [x for x in y_gt if x == 18]

        for k in range(len(sched)):
            if y_gt[k] == 18 and sched[k] > (len(sched) - len(unripe) - 1):
                sched[k] = 17
            sched_true[sched[k]][y_gt[k] - 1] += 1
            sched_students[sched[k]][y_stud[k] - 1] += 1
            sched_heuristic[sched[k]][y_heu[k] - 1] += 1
            occ_1[y_occ[k]][sched[k]] += 1
        sx = dx

'''
for i in range(len(real_tscheduling) - 1):
    print(f'\n {100*real_tscheduling[i]/real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where scheduled as {i + 1:.4f}')
print(f'\n {100*real_tscheduling[-1]/real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where UNRIPE')

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

w = Counter([item for sublist in pred.tolist() for item in sublist])
plt.bar(w.keys(), w.values(), width=0.0001)
#plt.savefig('testEasiness.png')

w = Counter([item for sublist in tbatch.y.tolist() for item in sublist])
plt.bar(w.keys(), w.values(), width=0.0001)
plt.title('Strawberry test easiness score. Orange: gt. Blue: predicted')
plt.savefig('truetestEasiness.png')
'''
# Generating data for the heat map
data = sched_true
# Function to show the heat map
fig, ax = plt.subplots()
plt.imshow(data)

# Adding details to the plot
plt.title("Number of strawberries")
plt.ylabel('predicted_scheduling')
plt.xlabel('easiness_scheduling')

# Adding a color bar to the plot
plt.colorbar()
for i in range(len(sched_true)):
    for j in range(len(sched_true)):
        text = ax.text(j, i, int(sched_true[i, j]),
                       ha="center", va="center", color="w")
# Displaying the plot
plt.savefig('heatmapEasiness.png')

# Generating data for the heat map HEURISTIC
data = sched_heuristic
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Number of strawberries")
plt.ylabel('scheduling order predicted')
plt.xlabel('heuristic_scheduling')
plt.colorbar()
for i in range(len(sched_heuristic)):
    for j in range(len(sched_heuristic)):
        text = ax.text(j, i, int(sched_heuristic[i, j]),
                       ha="center", va="center", color="w")

# Displaying the plot
plt.savefig('heatmapHeuristic.png')

# Generating data for the heat map STUDENTS
data = sched_students
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Number of strawberries")
plt.ylabel('scheduling order predicted')
plt.xlabel('students_scheduling')
plt.colorbar()
for i in range(len(sched_students)):
    for j in range(len(sched_students)):
        text = ax.text(j, i, int(sched_students[i, j]),
                       ha="center", va="center", color="w")

# Displaying the plot
plt.savefig('heatmapStudents.png')

# Generating data for the heat map STUDENTS
data = occ_1
fig, ax = plt.subplots()
plt.imshow(data)
# Adding details to the plot
plt.title("Occlusion property")
plt.xlabel('scheduling order predicted')
plt.ylabel('occlusion label')
# Show all ticks and label them with the respective list entries
occlusions = ['NON', ' BY LEAF', 'BY BERRY', 'UNRIPE']
ax.set_yticks(np.arange(len(occlusions)), labels=occlusions)

for i in range(len(occ_1)):
    for j in range(len(occ_1[i])):
        text = ax.text(j, i, int(occ_1[i, j]),
                       ha="center", va="center", color="w")

# Displaying the plot
plt.savefig('heatmapOcclusions.png')
'''