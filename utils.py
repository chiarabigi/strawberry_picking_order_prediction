import numpy as np
import torch
import json
from scipy.spatial import distance
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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

def get_sched(caption):
    sched = []
    for i in range(len(caption)):
        sched.append(int(caption[i].split(',')[-1]))
    return sched

## For annotations
def overlap(bbox1, bbox2):
    x1c = bbox1[0] + bbox1[2] / 2
    y1c = bbox1[1] + bbox1[3] / 2
    x2c = bbox2[0] + bbox2[2] / 2
    y2c = bbox2[1] + bbox2[3] / 2
    L = distance.euclidean((x1c, y1c), (x2c, y2c))
    if L < 30:
        return True
    else:
        return False

def true_unripe(boxes, ripe):
    unripe = []
    for i in range(len(boxes)):
        box = boxes[i]
        xmin = float(box[0])
        ymin = float(box[1])
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        box1 = [xmin, ymin, w, h]
        add = True
        for box2 in ripe:
            if overlap(box1, box2):
                add = False
        if add:
            unripe.append(box1)
    return unripe

def get_xy(boxes):
    xy = []
    for i in range(len(boxes)):
        bbox = boxes[i]
        xy.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])
    return xy

def get_patches(xy, d):
    # to compress image patches:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    xy = torch.tensor(xy)
    x, y = xy.T
    width, height = d.size
    d = transforms.ToTensor()(d)
    m = 112  # m = 1 means a patch of 3x3 centered around given pixel location
    for p in range(len(x)):
        if x[p] + m >= width:
            x[p] = width - m
        elif x[p] - m <= 0:
            x[p] = m
        if y[p] + m >= height:
            y[p] = height - m
        elif y[p] - m <= 0:
            y[p] = m
    o = torch.stack([d[:, yi - m: yi + m, xi - m: xi + m] for xi, yi in zip(x, y)])
    patches = []
    for a in range(len(o)):
        model.eval()
        compress = model.avgpool(model.features(o[a].unsqueeze(0))).squeeze(0).squeeze(-1).squeeze(-1)
        patches.append(compress.tolist())
    return

def get_info(str_ann, occ_ann):
    str_info = []
    for rc in range(len(str_ann)):
        xrmin = str_ann[rc][0]
        yrmin = str_ann[rc][1]
        xrmax = str_ann[rc][0] + str_ann[rc][2]
        yrmax = str_ann[rc][1] + str_ann[rc][3]
        xrc = str_ann[rc][0] + str_ann[rc][2] / 2
        yrc = str_ann[rc][1] + str_ann[rc][3] / 2
        wr2 = str_ann[rc][2] / 2
        area = str_ann[rc][2] * str_ann[rc][3]
        str_info.append({
            'xmin': xrmin,
            'ymin': yrmin,
            'xmax': xrmax,
            'ymax': yrmax,
            'xc': xrc,
            'yc': yrc,
            'w_half': wr2,
            'area': area,
            'occlusion': occ_ann[rc],
            'occlusion_by_berry%': 0
        })
    return str_info


def min_str_dist(all_strawberries, check_berry_occlusion):

    all_min_dist = []
    all_min_edges = []
    all_feats = []
    all_edges = []
    all_dist = {
        'dist': all_feats,
        'min_dist': all_min_dist,
        'edges': all_edges,
        'min_edges': all_min_edges
    }
    for i in range(len(all_strawberries)):

        if check_berry_occlusion:
            xrmin = all_strawberries[i]['xmin']
            yrmin = all_strawberries[i]['ymin']
            xrmax = all_strawberries[i]['xmax']
            yrmax = all_strawberries[i]['ymax']
            arear = all_strawberries[i]['area']
        xrc = all_strawberries[i]['xc']
        yrc = all_strawberries[i]['yc']
        wr2 = all_strawberries[i]['w_half']

        dist = []
        edges = []
        condition = [j for j in range(len(all_strawberries)) if j != i]
        for j in condition:
            if check_berry_occlusion:
                all_edges = edges
            if [i, j] not in all_edges:
                if check_berry_occlusion:
                    xomin = all_strawberries[j]['xmin']
                    yomin = all_strawberries[j]['ymin']
                    xomax = all_strawberries[j]['xmax']
                    yomax = all_strawberries[j]['ymax']
                    areao = all_strawberries[j]['area']
                xoc = all_strawberries[j]['xc']
                yoc = all_strawberries[j]['yc']
                wo2 = all_strawberries[j]['w_half']

                ip = abs(distance.euclidean((xrc, yrc), (xoc, yoc)))
                x = abs(float(xoc - xrc))

                ipr = 2 * wr2
                if x != 0:
                    cosalpha = x / ip
                    if abs(cosalpha) > 1:
                        print(0)
                    ipr = abs(wr2 / cosalpha)
                    ipo = abs(wo2 / cosalpha)
                if ipr >= 2 * wr2:
                    y = abs(float(yoc - yrc))
                    sinaplha = y / ip
                    if abs(sinaplha) > 1:
                        print(0)
                    ipr = abs(wr2 / sinaplha)
                    ipo = abs(wo2 / sinaplha)

                box_dist = ip - ipr - ipo  # distance between boxes

                # Is this strawberry occluded by another one?
                if check_berry_occlusion:
                    if box_dist < 0 and (ipr != wr2 and ipo != wo2):
                        dx = min(xrmax, xomax) - max(xrmin, xomin)
                        dy = min(yrmax, yomax) - max(yrmin, yomin)
                        if (dx < 0) or (dy < 0):  # corners don't overlap
                            if xomin < xrmin and yomin > yrmin:  # overlapping on the left
                                dx = xomax - xrmin
                                dy = yomax - yomin
                            elif xomin > xrmin and yomin < yrmin:  # overlapping on the bottom
                                dx = xomax - xomin
                                dy = yomax - yrmin
                            elif xomax < xrmax and yomax > yrmax:  # overlapping on the top
                                dx = xomax - xomin
                                dy = yrmax - yomax
                            elif xomax > xrmax and yomax < yrmax:  # overlapping on the right
                                dx = xrmax - xomin
                                dy = yomax - yomin
                            else:  # one inside the other
                                dx = xomax - xomin
                                dy = yomax - yomin
                        overlap_area = abs(dx) * abs(dy)
                        occlusion_fraction = overlap_area / arear
                        if occlusion_fraction < 1:  # I shall never know why this happens
                            all_strawberries[i]['occlusion'] = 4
                            all_strawberries[i]['occlusion_by_berry%'] = occlusion_fraction

                dist.append(abs(box_dist))
                edges.append([[i, j], [j, i]])

                all_feats += [abs(box_dist), abs(box_dist)]
                all_edges += [[i, j], [j, i]]

        if len(dist) > 0:
            all_min_dist += [abs(min(dist))]
            all_min_edges += edges[dist.index(min(dist))]

    ''''''
    max_dist = 1245
    all_min_dist = [x / max_dist for x in all_min_dist]
    all_feats = [x / max_dist for x in all_feats]
    if check_berry_occlusion:
        for b in all_strawberries:
            b['xmin'] = b['xmin'] / max_dist
            b['ymin'] = b['ymin'] / max_dist

    return all_dist


def get_dist_score(all_ripe_min_dist, diag):
    dist_score = []
    if len(all_ripe_min_dist) == 1:
        dist_score.append(1)
    else:
        for d in range(len(all_ripe_min_dist)):
            dist_score.append(1245 * all_ripe_min_dist[d] / diag)
    return dist_score

def get_occ_score(ripe_info):
    occ_score = []
    for r in range(len(ripe_info)):
        occ = ripe_info[r]['occlusion']
        if occ == 1 or occ == 3:  # non occluded OR occluding
            occ_score.append(1)
        elif occ == 0 or occ == 2:  # occluded by leaf OR occluding and occluded by leaf
            occ_score.append(0.6)
        else:  # occluded by berry
            occ_score.append(0.5 * (1 - ripe_info[r]['occlusion_by_berry%']))
    return occ_score

def update_occ(ripe_info):
    try:
        infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
    except IndexError:
        print(0)
    occ_updated = infoT['occlusion']
    for o in range(len(occ_updated)):
        if occ_updated[o] % 2 != 0:
            occ_updated[o] = 0  # non occluded
        elif occ_updated[o] == 4:
            occ_updated[o] = 2  # occluded by berry
        else:
            occ_updated[o] = 1  # occluded by leaf
    return occ_updated

def heuristic_sched(all_ripe_min_dist, occ_ann):
    neither = []
    occluding = []
    occluded_occluding = []
    occluded = []
    for k in range(len(all_ripe_min_dist)):
        if occ_ann[k] == 3:
            neither.append(all_ripe_min_dist[k])
        elif occ_ann[k] == 1:
            occluding.append(all_ripe_min_dist[k])
        elif occ_ann[k] == 2:
            occluded_occluding.append(all_ripe_min_dist[k])
        elif occ_ann[k] == 0:
            occluded.append(all_ripe_min_dist[k])

    neither_sorted_indices = sorted(range(len(neither)), reverse=True, key=lambda k: neither[k])
    occluding_sorted_indices = [x + len(neither) for x in
                                sorted(range(len(occluding)), reverse=True, key=lambda k: occluding[k])]
    occluded_occluding_sorted_indices = [x + len(neither) + len(occluding) for x in
                                         sorted(range(len(occluded_occluding)), reverse=True,
                                                key=lambda k: occluded_occluding[k])]
    occluded_sorted_indices = [x + len(neither) + len(occluding) + len(occluded_occluding_sorted_indices) for x in
                               sorted(range(len(occluded)), reverse=True, key=lambda k: occluded[k])]

    scheduling_script1 = neither_sorted_indices + occluding_sorted_indices + occluded_occluding_sorted_indices + occluded_sorted_indices
    scheduling_script1 = [x + 1 for x in scheduling_script1]

    return scheduling_script1

## For metrics

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




def get_single_out(batches, idx, sx):
    dx = sx
    found = False
    while not found and dx < len(batches):
        if int(batches[dx]) == idx + 1:
            found = True
        else:
            dx += 1

    return dx


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
                        try:
                            index = all_edge_indices.index(lasts)
                        except ValueError:
                            print(0)

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


def distances(ripe_ann):
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

    # max of min distance:
    all_dist = min_str_dist(ripe_info, False)
    all_dist['min_dist'] = [[x, x] for x in all_dist['min_dist']]
    all_dist['min_dist'] = [item for sublist in all_dist['min_dist'] for item in sublist]

    return all_dist['dist'], all_dist['edges'], all_dist['min_dist'], all_dist['min_edges']