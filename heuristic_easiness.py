import json
import random
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from collections import Counter
from utils import get_single_out, true_unripe, get_info, min_str_dist, get_dist_score, get_occ_score, update_occ, heuristic_sched, get_sched, get_patches

base_path = '/home/chiara/strawberry_picking_order_prediction/'
img_path = '/home/chiara/DATASETS/images/'   # images are to be downloaded

# unripe info
unripe_path = base_path + 'dataset/unripe.json'  # obtained with detectron2 ran on GPU
with open(unripe_path) as f:
    unripe_annT = json.load(f)
unripe_ann = {k: [dic[k] for dic in unripe_annT] for k in unripe_annT[0]}
easy_tot = []
w = Counter([])
phases = ['train', 'val', 'test']
for phase in phases:
    easy = []
    filepath = base_path +  'scheduling/data_{}/raw/gnnann.json'.format(phase)
    gnnann = []
    json_path = base_path + 'dataset/instances_{}.json'.format(phase)
    with open(json_path) as f:
        json_file = json.load(f)
    imagesT = json_file['images']
    images = {k: [dic[k] for dic in imagesT] for k in imagesT[0]}
    annsT = json_file['annotations']
    anns = {k: [dic[k] for dic in annsT] for k in annsT[0]}
    anns_sched = get_sched(anns['caption'])

    sx = 0
    for i in range(len(images['id'])):
        filename = images['file_name'][i]
        d = Image.open(img_path + filename)
        width, height = d.size
        diag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))

        dx = get_single_out(anns['image_id'], i, sx)

        ripe = anns['bbox'][sx:dx]
        occ = anns['category_id'][sx:dx]
        sched = anns_sched[sx:dx]
        tot_unripe = unripe_ann['bboxes'][i][sx:dx]
        sx = dx
        if len(tot_unripe) > 0:
            unripe = true_unripe(tot_unripe, ripe, diag)
            occ.extend([3] * len(unripe))
            unripe_info = get_info(unripe, occ)
        else:
            unripe_info = []
            unripe = []
        ripe_info = get_info(ripe, occ)
        len_ripe_info = len(ripe_info)

        ripe_info.extend(unripe_info)
        min_dist_ripe = min_str_dist(ripe_info, True)['min_dist'][:len_ripe_info]
        ripe_infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
        xy = list(map(list, zip(*[[int(x) for x in ripe_infoT['xc']], [int(y) for y in ripe_infoT['yc']]])))
        ripe_info = ripe_info[:len_ripe_info]

        dist_score = get_dist_score(min_dist_ripe, diag)
        occ_score = get_occ_score(ripe_info)

        easiness = [dist_score[e] * occ_score[e] for e in range(len(dist_score))]

        ''''''
        # balance scores
        distribution = w.most_common()
        easyr = [round(x, 4) for x in easiness]
        elem = [[y for y in range(len(easyr)) if (distribution[x][0] == easyr[y] and distribution[x][1] > 2)] for x in range(len(distribution))]
        elem = [item for sublist in elem for item in sublist]
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

        easy += [round(x, 4) for x in easiness]
        w = Counter(easy)

        scheduling_easiness = sorted(range(len(easiness)), reverse=True, key=lambda k: easiness[k])
        scheduling_easiness = [x + 1 for x in scheduling_easiness]

        scheduling_heuristic = heuristic_sched(min_dist_ripe, occ)

        occ_ann = update_occ(ripe_info)

        # scale bbox
        ripe = [[x / diag for x in r] for r in ripe]
        unripe = [[x / diag for x in u] for u in unripe]

        scheduling_easiness.extend([18] * len(unripe))
        easiness.extend([0] * len(unripe))
        occ_score.extend([1] * len(unripe))
        occ_ann.extend([3] * len(unripe))
        scheduling_heuristic.extend([18] * len(unripe))
        sched.extend([18] * len(unripe))
        ripeness = [1] * len(ripe) + [0] * len(unripe)
        boxes = ripe
        boxes.extend(unripe)

        # shuffle
        order = np.arange(0, len(boxes))
        random.shuffle(order)
        boxes = [boxes[j] for j in order]
        ripeness = [ripeness[j] for j in order]
        scheduling_easiness = [scheduling_easiness[k] for k in order]
        sched = [sched[k] for k in order]
        scheduling_heuristic = [scheduling_heuristic[k] for k in order]
        occ_ann = [occ_ann[h] for h in order]
        occ_score = [occ_score[h] for h in order]
        easiness = [easiness[h] for h in order]
        xy = [xy[n] for n in order]

        # patches
        # patches = get_patches(xy, d)

        gnnann.append({
            'img_ann': boxes,
            'sc_ann': scheduling_easiness,
            'students_sc_ann': sched,
            'heuristic_sc_ann': scheduling_heuristic,
            'occ_ann': occ_ann,
            'occ_score': occ_score,
            'easiness': easiness,
            # 'patches': patches,
            'ripeness': ripeness
        })

    one = 1
    w = Counter(easy)
    plt.bar(w.keys(), w.values(), width=0.001)
    plt.savefig('barEasiness_distributed_{}.png'.format(phase))
    easy_tot += easy

    ''''''
    save_path = base_path + 'dataset/data_{}/raw/gnnann.json'.format(phase)
    with open(save_path, 'w') as f:
        json.dump(gnnann, f)
    print(phase + str(len(gnnann)))


