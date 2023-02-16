import json
from scipy.spatial import distance
import copy
from PIL import Image
import math


oldS = [0] * 18
newS = [0] * 18
easyS = [0] * 18
boxes = 0
img_path = '/home/chiara/DATASETS/images/'
phases = ['train', 'val', 'test']
for phase in phases:
    anpath = '/home/chiara/strawberry_picking_order_prediction/dataset/scheduling/data_{}/raw/gnnann.json'.format(phase)
    with open(anpath) as f:
        anns = json.load(f)


    new_gnnann = []
    for g in range(len(anns)):

        ripe_ann = anns[g]['img_ann']
        unripe_ann = anns[g]['unripe']
        occ_ann = anns[g]['occ_ann']  # [:len(ripe_ann)]
        sc_ann = anns[g]['sc_ann']  # [:len(ripe_ann)]
        filename = anns[g]['filename']

        # sc_ann = sc_ann[:len(ripe_ann)]
        for i in range(len(sc_ann)):
            oldS[sc_ann[i] - 1] += 1
        boxes += len(ripe_ann)

        ripe_info = []
        for rc in range(len(ripe_ann)):
            xrmin = ripe_ann[rc][0]
            yrmin = ripe_ann[rc][1]
            xrmax = ripe_ann[rc][0] + ripe_ann[rc][2]
            yrmax = ripe_ann[rc][1] + ripe_ann[rc][3]
            xrc = ripe_ann[rc][0] + ripe_ann[rc][2] / 2
            yrc = ripe_ann[rc][1] + ripe_ann[rc][3] / 2
            wr2 = ripe_ann[rc][2] / 2
            area = ripe_ann[rc][2] * ripe_ann[rc][3]
            ripe_info.append({
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

        unripe_info = []
        for uc in range(len(unripe_ann)):
            xumin = unripe_ann[uc][0]
            yumin = unripe_ann[uc][1]
            xumax = unripe_ann[uc][0] + unripe_ann[uc][2]
            yumax = unripe_ann[uc][1] + unripe_ann[uc][3]
            xuc = unripe_ann[uc][0] + unripe_ann[uc][2] / 2
            yuc = unripe_ann[uc][1] + unripe_ann[uc][3] / 2
            wu2 = unripe_ann[uc][2] / 2
            area = unripe_ann[uc][2] * unripe_ann[uc][3]
            unripe_info.append({
                'xmin': xumin,
                'ymin': yumin,
                'xmax': xumax,
                'ymax': yumax,
                'xc': xuc,
                'yc': yuc,
                'w_half': wu2,
                'area': area,
                'occlusion': occ_ann[uc],
                'occlusion_by_berry%': 0
            })

        # max of min distance:

        all_ripe_min_dist = []
        all_ripe_min_edges = []
        for i in range(len(ripe_ann)):

            xrmin = ripe_info[i]['xmin']
            yrmin = ripe_info[i]['ymin']
            xrmax = ripe_info[i]['xmax']
            yrmax = ripe_info[i]['ymax']
            xrc = ripe_info[i]['xc']
            yrc = ripe_info[i]['yc']
            wr2 = ripe_info[i]['w_half']
            arear = ripe_info[i]['area']

            all_strawberries = copy.copy(ripe_info)  # I may have changed ripe_info
            all_strawberries.extend(unripe_info)

            dist = []
            edges = []
            condition = [j for j in range(len(all_strawberries)) if j != i]
            for j in condition:
                xomin = all_strawberries[j]['xmin']
                yomin = all_strawberries[j]['ymin']
                xomax = all_strawberries[j]['xmax']
                yomax = all_strawberries[j]['ymax']
                xoc = all_strawberries[j]['xc']
                yoc = all_strawberries[j]['yc']
                wo2 = all_strawberries[j]['w_half']
                areao = all_strawberries[j]['area']

                ip = abs(distance.euclidean((xrc, yrc), (xoc, yoc)))
                x = abs(float(xoc - xrc))
                try:
                    cosalpha = x / ip
                    if abs(cosalpha) > 1:
                        print(0)
                    ipr = abs(wr2 / cosalpha)
                    ipo = abs(wo2 / cosalpha)
                except ZeroDivisionError:
                    y = abs(float(yoc - yrc))
                    sinaplha = y / ip
                    if abs(sinaplha) > 1:
                        print(0)
                    ipr = abs(wr2 / sinaplha)
                    ipo = abs(wo2 / sinaplha)

                box_dist = ip - ipr - ipo  # distance between boxes
                edges.append([i, j])

                # Is this strawberry occluded by another one?
                if box_dist < 0 and (ipr != wr2 and ipo != wo2):
                    dx = min(xrmax, xomax) - max(xrmin, xomin)
                    dy = min(yrmax, yomax) - max(yrmin, yomin)
                    if (dx < 0) or (dy < 0):   # corners don't overlap
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
                            dx = xrmax -xomin
                            dy = yomax - yomin
                        else:  # one inside the other
                            dx = xomax - xomin
                            dy = yomax - yomin
                    overlap_area = abs(dx) * abs(dy)
                    occlusion_fraction = overlap_area / arear
                    if occlusion_fraction < 1:  # I shall never know why this happens
                        ripe_info[i]['occlusion'] = 4
                        ripe_info[i]['occlusion_by_berry%'] = occlusion_fraction

                dist.append(abs(box_dist))

            all_ripe_min_dist.append(min(dist))
            all_ripe_min_edges.append(edges[dist.index(min(dist))])

        # easiness:

        a1, a2 = Image.open(img_path + filename).size
        halph_diag = math.sqrt(math.pow(a1, 2) + math.pow(a2, 2)) / 2
        dist_score = []
        if len(all_ripe_min_dist) == 1:
            dist_score.append(1)
        else:
            for d in range(len(all_ripe_min_dist)):
                dist_score.append(all_ripe_min_dist[d] / halph_diag)

        occ_score = []
        for r in range(len(ripe_info)):
            occ = ripe_info[r]['occlusion']
            if occ == 1 or occ == 3:  # non occluded OR occluding
                occ_score.append(1)
            elif occ == 0 or occ == 2:  # occluded by leaf OR occluding and occluded by leaf
                occ_score.append(0.9)
            else:  # occluded by berry
                occ_score.append(0.7 * (1 - ripe_info[r]['occlusion_by_berry%']))

        easiness = [dist_score[e] * occ_score[e] for e in range(len(dist_score))]

        scheduling = sorted(range(len(easiness)), reverse=True, key=lambda k: easiness[k])
        scheduling = [x + 1 for x in scheduling]

        for i in range(len(scheduling)):
            easyS[scheduling[i] - 1] += 1

        scheduling.extend([18] * len(unripe_ann))
        easiness.extend([0] * len(unripe_ann))
        occ_score.extend([1] * len(unripe_ann))

        infoT = {k: [dic[k] for dic in ripe_info] for k in ripe_info[0]}
        occ_updated = infoT['occlusion']
        occ_updated.extend([3] * len(unripe_ann))

        # scheduling that prefers isolated non occluded strawberries

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

        for i in range(len(scheduling_script1)):
            newS[scheduling_script1[i] - 1] += 1

        new_gnnann.append({
            'img_ann': anns[g]['img_ann'],
            'sc_ann': scheduling,  # easiness
            'students_sc_ann': anns[g]['sc_ann'],
            'heuristic_sc_ann': scheduling_script1,
            'occ_ann': occ_updated,
            'occ_score': occ_score,
            'easiness': easiness,
            # 'patches': anns[g]['patches'],
            'unripe': anns[g]['unripe']
        })

    ''''''
    save_path = '/home/chiara/strawberry_picking_order_prediction/dataset/easiness/data_{}/raw/gnnann.json'.format(phase)
    with open(save_path, 'w') as f:
        json.dump(new_gnnann, f)
    print(phase + str(len(new_gnnann)))

print('students scheduling', oldS)
print('script1 scheduling', newS)
print('easiness scheduling', easyS)
print('% of similarity btw script1 and students', [100 * (newS[i] - abs(newS[i] - oldS[i])) / newS[i] for i in range(len(newS) - 1)])
print('% of similarity btw easiness and students', [100 * (easyS[i] - abs(easyS[i] - oldS[i])) / easyS[i] for i in range(len(easyS) - 1)])
print('% of similarity btw easiness and script1', [100 * (easyS[i] - abs(easyS[i] - newS[i])) / easyS[i] for i in range(len(easyS) - 1)])
