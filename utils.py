import numpy as np
import torch
import json
from scipy.spatial import distance
import copy


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
    This is the worst written function ever. Computationally, it makes me cry
    '''
    answer = True
    stop = False
    if len(edge_indices) > 1:
        while not stop:
            for j in range(len(edge_indices)):
                perimeter = edge_indices[j]
                for h in range(len(edge_indices)):  # I check it twice because I couldn't think of anything best
                    for i in range(len(edge_indices)):
                        if edge_indices[i] != edge_indices[j]:
                            if (edge_indices[i][0] in perimeter) or (edge_indices[i][-1] in perimeter):
                                perimeter = list(set(perimeter).symmetric_difference(set(edge_indices[i])))
                    if len(perimeter) == 0:
                        answer = False
                        stop = True
            stop = True
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

    return edge_feats, edge_indices


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


def distances(ripe_ann, unripe_ann):
    ripe_info = []
    for rc in range(len(ripe_ann)):
        xrc = ripe_ann[rc][0] + ripe_ann[rc][2] / 2
        yrc = ripe_ann[rc][1] + ripe_ann[rc][3] / 2
        wr2 = ripe_ann[rc][2] / 2
        ripe_info.append({
            'xc': xrc,
            'yc': yrc,
            'w_half': wr2
        })

    unripe_info = []
    for uc in range(len(unripe_ann)):
        xuc = unripe_ann[uc][0] + unripe_ann[uc][2] / 2
        yuc = unripe_ann[uc][1] + unripe_ann[uc][3] / 2
        wu2 = unripe_ann[uc][2] / 2
        unripe_info.append({
            'xc': xuc,
            'yc': yuc,
            'w_half': wu2,
        })

    # max of min distance:

    all_min_dist = []
    all_min_edges = []
    all_feats = []
    all_edges = []
    all_strawberries = copy.copy(ripe_info)  # I may have changed ripe_info
    all_strawberries.extend(unripe_info)
    for i in range(len(all_strawberries)):

        xrc = all_strawberries[i]['xc']
        yrc = all_strawberries[i]['yc']
        wr2 = all_strawberries[i]['w_half']

        dist = []
        edges = []
        # Compute all possible edges (sides + diagonals), ONCE
        condition = [j for j in range(len(all_strawberries)) if j != i]
        for j in condition:
            if [i, j] not in all_edges:
                xoc = all_strawberries[j]['xc']
                yoc = all_strawberries[j]['yc']
                wo2 = all_strawberries[j]['w_half']

                ip = distance.euclidean((xrc, yrc), (xoc, yoc))
                x = abs(float(xoc - xrc))
                try:
                    cosalpha = ip / x
                    ipr = wr2 * cosalpha
                    ior = wo2 * cosalpha
                except ZeroDivisionError:
                    y = abs(float(yoc - yrc))
                    sinaplha = ip / y
                    ipr = wr2 * sinaplha
                    ior = wo2 * sinaplha

                box_dist = ip - ipr - ior  # distance between boxes
                dist.append(box_dist)
                edges.append([[i, j], [j, i]])

                all_feats += [abs(box_dist), abs(box_dist)]
                all_edges += [[i, j], [j, i]]

        if len(dist) > 0:
            all_min_dist += [abs(min(dist)), abs(min(dist))]
            all_min_edges += edges[dist.index(min(dist))]

    return all_feats, all_edges, all_min_dist, all_min_edges