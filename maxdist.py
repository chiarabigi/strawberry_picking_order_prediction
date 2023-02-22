import json
from utils import get_single_out, true_unripe, get_info, min_str_dist

base_path = '/home/chiara/strawberry_picking_order_prediction/'  # images are to be downloaded

# unripe info
unripe_path = base_path + 'dataset/unripe.json'  # obtained with detectron2 ran on GPU
with open(unripe_path) as f:
    unripe_annT = json.load(f)
unripe_ann = {k: [dic[k] for dic in unripe_annT] for k in unripe_annT[0]}

maxd = 0
phases = ['train', 'val', 'test']
for phase in phases:
    json_path = base_path + 'dataset/isamesize_{}.json'.format(phase)
    with open(json_path) as f:
        json_file = json.load(f)
    imagesT = json_file['images']
    images = {k: [dic[k] for dic in imagesT] for k in imagesT[0]}
    annsT = json_file['annotations']
    anns = {k: [dic[k] for dic in annsT] for k in annsT[0]}

    sx = 0
    for i in range(len(images['id'])):
        filename = images['file_name'][i]
        dx = get_single_out(anns['image_id'], i, sx)

        ripe = anns['bbox'][sx:dx]
        occ = anns['category_id'][sx:dx]
        tot_unripe = [unripe_ann['bboxes'][x] for x in range(len(unripe_ann['bboxes']))
                      if unripe_ann['file_name'][x] == filename.split('_')[-1]][0]
        sx = dx
        if len(tot_unripe) > 0:
            unripe = true_unripe(tot_unripe, ripe)
            occ.extend([3] * len(unripe))
            unripe_info = get_info(unripe, occ)
        else:
            unripe_info = []
            unripe = []
        ripe_info = get_info(ripe, occ)
        len_ripe_info = len(ripe_info)

        ripe_info.extend(unripe_info)
        all_dist = min_str_dist(ripe_info, True)
        if max(all_dist['dist']) > maxd:
            maxd = max(all_dist['dist'])

print(maxd)  # 1314.16032512918
