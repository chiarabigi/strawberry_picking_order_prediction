import os
import json
import torch
import random
import numpy as np
from PIL import Image
from utils.utils import get_info, update_occ, get_patches
from utils.edges import min_str_dist

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__)).strip('data_scripts')


def ann_to_gnnann(images, images_path):

    gnnann = []  # where to store each graph information

    for img in images:
        filename = img['file_name']

        # from the list with all the annotations, extract just the ones of the image in consideration
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
        occ_ann = [occ_ann[h] for h in order]
        occ_leaf = [occ_leaf[h] for h in order]
        occ_score = [occ_score[h] for h in order]
        xy = [xy[n] for n in order]
        min_dist = [min_dist[m] for m in order]

        # patches (not now, first let's obtain a good model without it)
        # d = Image.open(images_path)
        patches = []  # get_patches(xy, d)

        gnnann.append({
            'img_ann': coord,
            'min_dist': min_dist,
            'boxes': boxes,
            'patches': patches,
            'occ_ann': occ_ann,
            'occ_score': occ_score,
            'occ_leaf': occ_leaf,
            'target_vector': [],
            'success': 0
        })

    save_path = images_path.strip(images_path.split('/')[-1])
    save_path_folder = save_path + 'raw'
    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)

    save_path_json = save_path_folder + '/gnnann.json'
    with open(save_path_json, "w") as f:
        json_str = json.dumps(gnnann)
        f.write(json_str)
    return save_path
