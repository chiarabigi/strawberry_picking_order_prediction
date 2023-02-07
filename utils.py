import numpy as np
import torch
import json


def unite_infos(json_annotations_path, target):
    with open(json_annotations_path+'/raw/gnnann.json') as f:
        anns = json.load(f)
    anns = anns[0]
    target_vector = [False] * len(anns[0])
    target_vector[target] = True
    new_anns = [anns[0], anns[2], target_vector]
    json_annotations_path = '/home/chiara/SCHEDULING/GNN/experiment_test/raw/gnnann.json'
    with open(json_annotations_path, "w") as f:
        json_str = json.dumps(new_anns)
        f.write(json_str)

    return json_annotations_path


## For metrics

def get_single_out(batches, idx, sx):
    dx = sx
    found = False
    while not found and dx < len(batches):
        if int(batches[dx]) == idx + 1:
            found = True
        else:
            dx += 1

    return dx


class BatchAccuracy_success(torch.nn.Module):
    def __init__(self):
        super(BatchAccuracy_success, self).__init__()

    def forward(self, y_pred, y_true):
        corrects = 0
        for j in range(len(y_true)):
            if y_pred[j] > 0.5:
                if y_true[j] >= 0.9:
                    corrects += 1
            else:
                if y_true[j] < 0.5:
                    corrects += 1
        return corrects


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


## For Dataset


def would_not_close_circle(edge_indices):  # will this combination of edges form a closed line?
    '''
    Works like domino: if I have edge 1-2 and 2-3, my perimeter becomes 1-3
    If at the end I have two different extremities, it means I didn't form a closed line
    '''
    answer = True
    if len(edge_indices) > 1:
        perimeter = edge_indices[0]
        for j in range(len(edge_indices)):  # I check twice... computationally stupid, but once is not enough
            for i in range(len(edge_indices) - 1):
                if (edge_indices[i + 1][0] in perimeter) or (edge_indices[i + 1][-1] in perimeter):
                    perimeter = list(set(perimeter).symmetric_difference(set(edge_indices[i + 1])))
        if len(perimeter) == 0:
            answer = False

    return answer


def only_sides(all_edge_feats, all_edge_indices, box):
    higher_indicator = np.asarray(torch.sort(torch.tensor(all_edge_feats)).indices)

    if len(box) == 2:  # 2 nodes: I take the only edge there is
        edge_feats = np.asarray([all_edge_feats[0], all_edge_feats[0]])
        edge_indices = np.asarray([[0, 1], [1, 0]])
    else:  # I want just the "sides" of the "polygon", preferring the shortest edges
        edge_feats = []
        edge_indices = []
        half_edge_indices = []
        all_nodes = np.zeros([len(box), 2])  # to check that I have just two edges per node
        circle = []
        while np.count_nonzero(all_nodes) != 2 * len(box):
            for h in range(len(higher_indicator)):
                i = all_edge_indices[higher_indicator[h]][0]
                j = all_edge_indices[higher_indicator[h]][1]
                if (0 in all_nodes[i]) and (0 in all_nodes[j]):
                    if would_not_close_circle(half_edge_indices + [[i, j]]):
                        circle.append(all_edge_indices[higher_indicator[h]])
                        L = all_edge_feats[higher_indicator[h]]
                        edge_feats += [L, L]
                        edge_indices += [[i, j], [j, i]]
                        half_edge_indices.append([i, j])
                        zeroI = np.where(all_nodes[i] == 0)[0][0]
                        zeroJ = np.where(all_nodes[j] == 0)[0][0]
                        all_nodes[i][zeroI] = 1
                        all_nodes[j][zeroJ] = 1
            if len(edge_feats) == (len(box) - 1) * 2:  # Am I missing just the last edge? Let me "close the circle"
                lasts = []
                for z in range(len(all_nodes)):
                    if all_nodes[z][-1] == 0:
                        lasts.append(z)
                # print(lasts)
                # Let's not close the circle
                if len(box) != 3:  # If it's a triangle, I chose to have just two edges
                    edge_indices += [lasts, list(reversed(lasts))]
                    try:
                        index = all_edge_indices.index(list(reversed(lasts)))
                    except ValueError:
                        index = all_edge_indices.index(lasts)
                    edge_feats += [all_edge_feats[index], all_edge_feats[index]]
                all_nodes[lasts[0]][-1] = 1
                all_nodes[lasts[-1]][-1] = 1


def highests(list):  # I discovered that torch.sort does a similar thing and better
    vector = np.array(list)
    higher_indicator = np.zeros_like(vector)
    o = 1
    while np.count_nonzero(higher_indicator) != len(higher_indicator):
        for i in range(len(vector)):
            if vector[i] == np.max(vector) and vector[i]!=-1:
                higher_indicator[i] = o
                o += 1
                vector[i]=-1
    higher_indicator = higher_indicator - np.ones_like(higher_indicator)
    return higher_indicator.astype(int)
