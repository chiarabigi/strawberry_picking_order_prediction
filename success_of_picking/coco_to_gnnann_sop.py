import json
import os
from utils.utils import get_info, update_occ, get_patches
from utils.edges import min_str_dist

whole_path = ''  # path/to/dataset
occlusion_properties = ['occluded', 'occluding', 'occluded/occluding', 'neither']

# annotations and images have to be already split in train/test/val
phases = ['train', 'val', 'test']
for phase in phases:
    filepath = whole_path + 'data_{}/raw/gnnann.json'.format(phase)
    gnnann = []
    phase_path = whole_path + phase + '/annotations'
    all_files = os.listdir(phase_path)
    for a in all_files:
        json_path = os.path.join(phase_path, a)
        with open(json_path) as f:
            anns = json.load(f)
        bboxs = []
        occlusions = []
        target = []
        for i in range(len(anns['annotations'])):
            bboxs.append(anns['annotations'][i]['bbox'])
            occ = occlusion_properties.index(anns['annotations'][i]['occlusion'])
            occlusions.append(occ)
            target.append(anns['annotations'][i]['target'])

        ripe_info = get_info(bboxs, occlusions)

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
        boxes = bboxs

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

        gnnann.append([bboxs, occlusions, target, anns['success']])

    ''''''
    with open(filepath, 'w') as f:
        json.dump(gnnann, f)
    print(phase + str(len(gnnann)))
