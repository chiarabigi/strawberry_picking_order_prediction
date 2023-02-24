import json
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from utils.utils import get_single_out, true_unripe, get_info, get_dist_score, get_occ_score, update_occ, heuristic_sched, get_patches
from utils.edges import min_str_dist

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__)).strip('data_scripts')
img_path = '/home/chiara/DATASETS/images/'   # images are to be downloaded

# unripe info
unripe_path = base_path + '/dataset/unripe.json'  # obtained with detectron2
with open(unripe_path) as f:
    unripe_annT = json.load(f)  # list of dictionaries
unripe_ann = {k: [dic[k] for dic in unripe_annT] for k in unripe_annT[0]}  # dictionary of lists

phases = ['train', 'val', 'test']
for phase in phases:
    # w = Counter([])  # initialization for first iteration, if you want to have balanced easiness score values
    gnnann = []  # where to store each graph information

    # for the bars loss_plots:
    easyS = []
    easyP = []
    heuS = []
    stuS = []
    heuP = []
    stuP = []

    filepath = base_path + '/scheduling/data_{}/raw/gnnann.json'.format(phase)  # where to save the annotation
    json_path = base_path + '/dataset/instances_{}.json'.format(phase)
    with open(json_path) as f:
        json_file = json.load(f)  # annotations in COCO format
    imagesT = json_file['images']
    images = {k: [dic[k] for dic in imagesT] for k in imagesT[0]}
    annsT = json_file['annotations']
    anns = {k: [dic[k] for dic in annsT] for k in annsT[0]}

    sx = 0
    for i in range(len(images['id'])):
        filename = images['file_name'][i]

        # from the list with all the annotations, extract just the ones of the image in consideration
        dx = get_single_out(anns['image_id'], i, sx)
        ripe = anns['bbox'][sx:dx]
        occ = anns['category_id'][sx:dx]
        students_scheduling = anns['caption'][sx:dx]
        tot_unripe = [unripe_ann['bboxes'][x] for x in range(len(unripe_ann['bboxes']))
                      if unripe_ann['file_name'][x] == filename][0]
        sx = dx

        if len(tot_unripe) > 0:
            unripe = true_unripe(tot_unripe, ripe)
            occ.extend([3] * len(unripe))
            unripe_info = get_info(unripe, occ)
        else:
            unripe_info = []
            unripe = []
        ripe_info = get_info(ripe, occ)
        ripe_info.extend(unripe_info)

        min_dist = min_str_dist(ripe_info, True)['min_dist']
        min_dist_ripe = min_dist[:len(ripe)]

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
        ripeness = [1] * len(ripe) + [0] * len(unripe)
        easiness = [dist_score[e] * occ_sc[e] + 0.11467494382197191 for e in range(len(dist_score))]

        '''
        # balance scores: are there some scores you want to discard?
        distribution = w.most_common()
        easyr = [round(x, 4) for x in easiness]
        elem = [y for y in range(len(easyr))
                 if easyr[y] > 0.01]  # indices to remove
        #elem = [item for sublist in elem for item in sublist]
        if len(elem) >= len(easiness) - 1:  # don't add anything of this graph if you remain with one or no berry
            continue
        indices = sorted(elem, reverse=True)
        for idx in indices:  # remove information of strawberries
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

        # scheduling from easiness score: first to pick has highest score, and so on
        scheduling_easiness = sorted(range(len(easiness)), reverse=True, key=lambda k: easiness[k])
        scheduling_easiness = [x + 1 for x in scheduling_easiness]
        # get a vector of probabilities that sum to one. Highest probability belongs to highest score
        easiness_prob = [(len(scheduling_easiness) + 1 - x) / sum(scheduling_easiness) for x in scheduling_easiness]
        easyS += easiness  # save for plot
        easyP += easiness_prob

        # scheduling from heuristic min-max approach: first to pick is most isolated & non occluded strawberry
        scheduling_heuristic = heuristic_sched(min_dist, occ)
        scheduling_heuristic = sorted(range(len(scheduling_heuristic)), reverse=True, key=lambda k: scheduling_heuristic[k])
        scheduling_heuristic = [x + 1 for x in scheduling_heuristic]  # we want scheduling to start from 1, not 0
        # get score [0, 1]. 1 = first to be picked; 0 = unripe; the rest are equally distances
        heu_score = [(len(scheduling_heuristic) - x) / (len(scheduling_heuristic) - 1) for x in scheduling_heuristic]
        heu_prob = [(len(scheduling_heuristic) + 1 - x) / sum(scheduling_heuristic) for x in scheduling_heuristic]
        heuP += heu_prob  # save for plot
        heuS += heu_score

        # get score and probability of annotated scheduling as well
        students_scheduling.extend([len(ripe) + 1] * len(unripe))  # because there was no annotation for unripe berries
        stud_score = [(len(ripe) + 1 - x) / (len(ripe) + 1 - 1) for x in students_scheduling]
        stud_prob = [(len(ripe) + 1 - x) / sum(students_scheduling) for x in students_scheduling]
        stuP += stud_prob  # save for plot
        stuS += stud_score

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
        boxes.extend(unripe)

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
        students_scheduling = [students_scheduling[k] for k in order]
        scheduling_heuristic = [scheduling_heuristic[k] for k in order]
        easiness = [easiness[h] for h in order]
        stud_score = [stud_score[h] for h in order]
        heu_score = [heu_score[h] for h in order]
        easiness_prob = [easiness_prob[h] for h in order]
        stud_prob = [stud_prob[h] for h in order]
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
            'students_sc_ann': students_scheduling,
            'heuristic_sc_ann': scheduling_heuristic,
            'stud_score': stud_score,
            'heu_score': heu_score,
            'easiness_score': easiness,
            'stud_prob': stud_prob,
            'heu_prob': heu_prob,
            'easiness_prob': easiness_prob
        })

    ''''''  # save annotation of graphs
    save_path = base_path + '/dataset/data_{}/raw/gnnann.json'.format(phase)
    with open(save_path, 'w') as f:
        json.dump(gnnann, f)
    print(phase + str(len(gnnann)))

    # plot distribution of scores / probabilities. Blue: train, orange:val, green: test
    y = Counter(stuP)
    plt.figure(1)
    plt.bar(y.keys(), y.values(), width=0.01)

    y = Counter(stuS)
    plt.figure(2)
    plt.bar(y.keys(), y.values(), width=0.01)

    y = Counter(heuP)
    plt.figure(3)
    plt.bar(y.keys(), y.values(), width=0.01)

    y = Counter(heuS)
    plt.figure(4)
    plt.bar(y.keys(), y.values(), width=0.01)

    y = Counter(easyP)
    plt.figure(5)
    plt.bar(y.keys(), y.values(), width=0.01)

    y = Counter(easyS)
    plt.figure(6)
    plt.bar(y.keys(), y.values(), width=0.01)

plt.figure(1)
plt.title('Students probabilities distribution.')
plt.savefig(base_path + 'imgs/data_bars/barStudentsProb_traintestval.png')

plt.figure(2)
plt.title('Students score distribution.')
plt.savefig(base_path + 'imgs/data_bars/barStudentsScore_traintestval.png')

plt.figure(3)
plt.title('Heuristic min-max probabilities distribution.')
plt.savefig(base_path + 'imgs/data_bars/barHeuristicProb_traintestval.png')

plt.figure(4)
plt.title('Heuristic min-max score distribution.')
plt.savefig(base_path + 'imgs/data_bars/barHeuristicScore_traintestval.png')

plt.figure(5)
plt.title('Heuristic easiness  probabilities distribution.')
plt.savefig(base_path + 'imgs/data_bars/barEasinessProb_traintestval.png')

plt.figure(6)
plt.title('Heuristic easiness score distribution.')
plt.savefig(base_path + 'imgs/data_bars/barEasinessScore_traintestval.png')



# Here are some files that produced big easiness score. Can be used for data augmentation
# big_scores = ['424.png', '442.png', '627.png', '573.png', '2390.png', '500.png', '524.png', '790.png', '1585.png', '366.png', '1569.png', '609.png', '473.png', '204.png', '427.png', '1575.png', '1120.png', '1543.png', '279.png', '458.png', '512.png', '118.png', '586.png', '437.png', '568.png', '555.png', '402.png', '798.png', '506.png', '336.png', '1658.png', '675.png', '1508.png', '444.png', '481.png', '254.png', '1524.png', '523.png']
