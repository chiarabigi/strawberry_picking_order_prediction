import torch
import math
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def overlap(bbox1, bbox2):
    x1c = bbox1[0] + bbox1[2] / 2
    y1c = bbox1[1] + bbox1[3] / 2
    x2c = bbox2[0] + bbox2[2] / 2
    y2c = bbox2[1] + bbox2[3] / 2
    L = math.sqrt(math.pow(x1c - x2c, 2) + math.pow(y1c - y2c, 2))  # distance.euclidean((x1c, y1c), (x2c, y2c))
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


def get_dist_score(all_ripe_min_dist):
    dist_score = []
    if len(all_ripe_min_dist) == 1:
        dist_score.append(1)
    else:
        for d in range(len(all_ripe_min_dist)):
            dist_score.append(all_ripe_min_dist[d])
    return dist_score

def get_occ_score(ripe_info):
    occ_score = []
    for r in range(len(ripe_info)):
        occ = ripe_info[r]['occlusion']
        if occ == 1 or occ == 3:  # non occluded OR occluding
            occ_score.append(1)
        elif occ == 0 or occ == 2:  # occluded by leaf OR occluding and occluded by leaf
            occ_score.append(0.7)
        else:  # occluded by berry
            occ_score.append(0.7*(ripe_info[r]['occlusion_by_berry%']))  # % of free area
    return occ_score

def update_occ(ripe_info):
    infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
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


def get_single_out(batches, idx, sx):
    dx = sx
    found = False
    while not found and dx < len(batches):
        if int(batches[dx]) == idx + 1:
            found = True
        else:
            dx += 1

    return dx
