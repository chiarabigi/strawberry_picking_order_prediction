import os
import torch
import random
import numpy as np
from PIL import Image
from utils.utils import get_info, get_dist_score, get_occ_score, update_occ, heuristic_sched
from utils.edges import min_str_dist

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__)).strip('data_scripts')


def ann_to_gnnann(images):
    gnnann = []  # where to store each graph information

    for img in images:
        filename = img['file_name']

        # from the list with all the annotations, extract just the ones of the image in consideratio
        ripe = img['annotations']['bbox']
        occ = img['annotations']['occlusion']

        ripe_info = get_info(ripe, occ)

        min_dist = min_str_dist(ripe_info, True)['min_dist']

        ripe_infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
        # save x,y coordinates of the center to later extract patches of images around those
        xy = list(map(list, zip(*[[int(x) for x in ripe_infoT['xc']], [int(y) for y in ripe_infoT['yc']]])))

        # save coordinates of x,y min to use as node features
        coordT = [ripe_infoT['xmin'], ripe_infoT['ymin']]
        coord = [[coordT[0][i], coordT[1][i]] for i in range(len(coordT[0]))]
        # save percentage of berry occlusion to use as node feature
        occ_score = ripe_infoT['occlusion_by_berry%']

        # EASINESS SCORE
        dist_score = get_dist_score(min_dist)
        occ_sc = get_occ_score(ripe_info)
        ripeness = [1] * len(ripe)
        easiness = [dist_score[e] * occ_sc[e] + 0.11467494382197191 for e in range(len(dist_score))]

        # scheduling from easiness score: first to pick has highest score, and so on
        scheduling_easiness = sorted(range(len(easiness)), reverse=True, key=lambda k: easiness[k])
        scheduling_easiness = [x + 1 for x in scheduling_easiness]
        # get a vector of probabilities that sum to one. Highest probability belongs to highest score
        easiness_prob = [(len(scheduling_easiness) + 1 - x) / sum(scheduling_easiness) for x in scheduling_easiness]

        # scheduling from heuristic min-max approach: first to pick is most isolated & non occluded strawberry
        scheduling_heuristic = heuristic_sched(min_dist, occ)
        scheduling_heuristic = sorted(range(len(scheduling_heuristic)), reverse=True, key=lambda k: scheduling_heuristic[k])
        scheduling_heuristic = [x + 1 for x in scheduling_heuristic]  # we want scheduling to start from 1, not 0
        # get score [0, 1]. 1 = first to be picked; 0 = unripe; the rest are equally distances
        heu_score = [(len(scheduling_heuristic) - x) / (len(scheduling_heuristic) - 1) for x in scheduling_heuristic]
        heu_prob = [(len(scheduling_heuristic) + 1 - x) / sum(scheduling_heuristic) for x in scheduling_heuristic]

        # previous occlusion options:
        # 'occluded by leaf', 'occluding', 'occluded by leaf/occluding', 'non occluded', 'occluded by berry'
        # updated occlusion options: 'non occluded', 'occluded by leaf', 'occluded by berry'
        occ_ann = update_occ(ripe_info)
        # get binary information of occlusion by leaf
        occ_leaf = [1] * len(occ_ann)
        for x in range(len(occ_ann)):
            if occ_ann[x] != 1:
                occ_leaf[x] = 0

        # save bbox coordinate information to use as node features
        boxes = ripe

        # shuffle! Some students annotation are ordered, you don't want the model to learn first to pick = node number 1
        order = np.arange(0, len(coord))
        random.shuffle(order)
        boxes = [boxes[j] for j in order]
        coord = [coord[j] for j in order]
        ripeness = [ripeness[j] for j in order]
        occ_ann = [occ_ann[h] for h in order]
        occ_leaf = [occ_leaf[h] for h in order]
        occ_score = [occ_score[h] for h in order]
        xy = [xy[n] for n in order]
        min_dist = [min_dist[m] for m in order]
        scheduling_easiness = [scheduling_easiness[k] for k in order]
        scheduling_heuristic = [scheduling_heuristic[k] for k in order]
        easiness = [easiness[h] for h in order]
        heu_score = [heu_score[h] for h in order]
        easiness_prob = [easiness_prob[h] for h in order]
        heu_prob = [heu_prob[h] for h in order]

        # patches (not now, first let's obtain a good model without it)
        # d = Image.open(img_path +  filename)
        patches = []  # get_patches(xy, d)

        gnnann.append({
            'img_ann': coord,
            'min_dist': min_dist,
            'boxes': boxes,
            'ripeness': ripeness,
            'patches': patches,
            'occ_ann': occ_ann,
            'occ_score': occ_score,
            'occ_leaf': occ_leaf,
            'easiness_sc_ann': scheduling_easiness,
            'students_sc_ann': [],
            'heuristic_sc_ann': scheduling_heuristic,
            'stud_score': [],
            'heu_score': heu_score,
            'easiness_score': easiness,
            'stud_prob': [],
            'heu_prob': heu_prob,
            'easiness_prob': easiness_prob
        })

    return gnnann