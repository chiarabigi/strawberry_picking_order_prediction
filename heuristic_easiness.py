import json
import os
import torch
import random
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from collections import Counter
from utils import get_single_out, true_unripe, get_info, min_str_dist, get_dist_score, get_occ_score, update_occ, heuristic_sched, get_sched, get_patches

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

base_path = os.path.dirname(os.path.abspath(__file__))
img_path = '/home/chiara/DATASETS/images/'   # images are to be downloaded

# unripe info
big_dist = []
big_scores = []
unripes_tot = []
ripes_tot = []
unripe_path = base_path + '/dataset/unripe.json'  # obtained with detectron2 ran on GPU
with open(unripe_path) as f:
    unripe_annT = json.load(f)
unripe_ann = {k: [dic[k] for dic in unripe_annT] for k in unripe_annT[0]}
easy_tot = []
ueasy_tot = []
w = Counter([])
phases = ['train', 'val', 'test']
for phase in phases:
    unripes = []
    ripes = []
    easy = []
    ueasy = []
    filepath = base_path + '/scheduling/data_{}/raw/gnnann.json'.format(phase)
    gnnann = []
    json_path = base_path + '/dataset/isamesize_{}.json'.format(phase)
    with open(json_path) as f:
        json_file = json.load(f)
    imagesT = json_file['images']
    images = {k: [dic[k] for dic in imagesT] for k in imagesT[0]}
    annsT = json_file['annotations']
    anns = {k: [dic[k] for dic in annsT] for k in annsT[0]}

    sx = 0
    for i in range(len(images['id'])):
        filename = images['file_name'][i]
        d = Image.open(img_path + filename.split('_')[-1])
        width, height = d.size
        diag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))

        dx = get_single_out(anns['image_id'], i, sx)

        ripe = anns['bbox'][sx:dx]
        ripes += [len(ripe)]
        occ = anns['category_id'][sx:dx]
        sched = anns['scheduling'][sx:dx]
        tot_unripe = [unripe_ann['bboxes'][x] for x in range(len(unripe_ann['bboxes'])) if unripe_ann['file_name'][x] == filename.split('_')[-1]][0]
        sx = dx
        if len(tot_unripe) > 0:
            unripe = true_unripe(tot_unripe, ripe)
            occ.extend([3] * len(unripe))
            unripe_info = get_info(unripe, occ)
            unripes += [len(unripe)]
        else:
            unripe_info = []
            unripe = []
        ripe_info = get_info(ripe, occ)
        len_ripe_info = len(ripe_info)

        ripe_info.extend(unripe_info)
        min_dist = min_str_dist(ripe_info, True)['min_dist']
        min_dist_ripe = min_dist[:len_ripe_info]
        high_dist = [x for x in min_dist_ripe if x > 0.5]
        if len(high_dist) > 0:
            big_dist.append(filename.split('_')[-1])
        ripe_infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
        xy = list(map(list, zip(*[[int(x) for x in ripe_infoT['xc']], [int(y) for y in ripe_infoT['yc']]])))
        # ripe_info = ripe_info[:len_ripe_info]

        coordT = [ripe_infoT['xmin'], ripe_infoT['ymin']]
        coord = [[coordT[0][i], coordT[1][i]] for i in range(len(coordT[0]))]

        dist_score = get_dist_score(min_dist)
        occ_sc = get_occ_score(ripe_info)
        ripeness = [1] * len(ripe) + [0] * len(unripe)

        easiness = [dist_score[e] * occ_sc[e] + ripeness[e] * 0.5 for e in range(len(dist_score))]
        high_score = [x for x in easiness if x > 0.5]
        if len(high_score) > 0:
            big_scores.append(filename.split('_')[-1])
        occ_score = ripe_infoT['occlusion_by_berry%']

        '''
        # balance scores
        distribution = w.most_common()
        easyr = [round(x, 4) for x in easiness]
        elem = [y for y in range(len(easyr))
                 if easyr[y] > 0.01]
        #elem = [item for sublist in elem for item in sublist]
        if len(elem) >= len(easiness) - 1:
            continue
        indices = sorted(elem, reverse=True)
        for idx in indices:
            easiness.pop(idx)
            ripe.pop(idx)
            ripe_info.pop(idx)
            min_dist_ripe.pop(idx)
            occ.pop(idx)
            occ_score.pop(idx)
            dist_score.pop(idx)
            xy.pop(idx)
            sched.pop(idx)
            coord.pop(idx)'''

        # easiness = [(x + 1) / 2 for x in easiness]
        # easiness = [x * 100 for x in easiness]
        scheduling_easiness = sorted(range(len(easiness)), reverse=True, key=lambda k: easiness[k])
        scheduling_easiness = [x + 1 for x in scheduling_easiness]

        scheduling_heuristic = heuristic_sched(min_dist_ripe, occ)
        scheduling_heuristic = sorted(range(len(scheduling_heuristic)), reverse=True, key=lambda k: scheduling_heuristic[k])
        scheduling_heuristic = [x + 1 for x in scheduling_heuristic]

        sched = sorted(range(len(sched)), reverse=True, key=lambda k: sched[k])
        sched = [x + 1 for x in sched]

        occ_ann = update_occ(ripe_info)
        occ_leaf = [1] * len(occ_ann)
        for x in range(len(occ_ann)):
            if occ_ann[x] != 1:
                occ_leaf[x]= 0

        scheduling_easiness.extend([18] * len(unripe))
        # easiness.extend([0] * len(unripe))
        occ_ann.extend([0] * len(unripe))
        occ_leaf.extend([0] * len(unripe))
        scheduling_heuristic.extend([18] * len(unripe))
        sched.extend([18] * len(unripe))

        easy += easiness[:len_ripe_info]  # [round(x, 4) for x in easiness]
        ueasy += easiness[len_ripe_info:]
        w = Counter(easy)

        boxes = ripe
        boxes.extend(unripe)

        # shuffle
        order = np.arange(0, len(coord))
        random.shuffle(order)
        boxes = [boxes[j] for j in order]
        coord = [coord[j] for j in order]
        ripeness = [ripeness[j] for j in order]
        scheduling_easiness = [scheduling_easiness[k] for k in order]
        sched = [sched[k] for k in order]
        scheduling_heuristic = [scheduling_heuristic[k] for k in order]
        occ_ann = [occ_ann[h] for h in order]
        occ_leaf = [occ_leaf[h] for h in order]
        easiness = [easiness[h] for h in order]
        xy = [xy[n] for n in order]
        min_dist = [min_dist[m] for m in order]

        # patches
        if device.type == 'cpu':
            patches = []
        else:
            patches = []  # get_patches(xy, d)

        gnnann.append({
            'img_ann': coord,
            'min_dist': min_dist,
            'boxes': boxes,
            'sc_ann': scheduling_easiness,
            'students_sc_ann': sched,
            'heuristic_sc_ann': scheduling_heuristic,
            'occ_ann': occ_ann,
            'occ_score': occ_score,
            'occ_leaf': occ_score,
            'easiness': easiness,
            'patches': patches,
            'ripeness': ripeness
        })
    gnnannT = {k: [dic[k] for dic in gnnann] for k in gnnann[0]}
    maxbox = max([item for sublist in gnnannT['boxes'] for item in sublist])
    one = 1
    w = Counter(easy+ueasy)
    plt.bar(w.keys(), w.values(), width=0.001)
    plt.savefig('imgs/barEasiness_distributed_traintestval.png')
    easy_tot += easy
    ueasy_tot += ueasy
    unripes_tot.append(unripes)
    ripes_tot.append(ripes)

    ''''''
    save_path = base_path + '/dataset/data_{}/raw/gnnann.json'.format(phase)
    with open(save_path, 'w') as f:
        json.dump(gnnann, f)
    print(phase + str(len(gnnann)))  # train784, val123, test118

print(max(easy_tot))
print(min([x for x in easy_tot if x != 0]))
print(max(ueasy_tot))
print(min(ueasy_tot))
one = 1
# WITH MANY UNRIPES
# < 0.01 train299, val45, test44, 0.01
# =< 0.1: train407, val62, test66, 0.5623
# > 0.1: train87, val13, test14, 1.0

# FEW UNRIPES
# < 0.01 train151, val23, test25
# < 0.1 train384, val58, test61
# > 0.1 train784, val123, test118
