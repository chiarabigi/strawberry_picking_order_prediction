import numpy as np
import torch
import math


def min_str_dist(all_strawberries, check_berry_occlusion):

    all_min_dist = []
    all_min_edges = []
    all_feats = []
    all_edges = []
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

                ip = math.sqrt(math.pow(xrc - xoc, 2) + math.pow(yrc - yoc, 2))  # abs(distance.euclidean((xrc, yrc), (xoc, yoc)))
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
                            all_strawberries[i]['occlusion_by_berry%'] = 1 - occlusion_fraction

                dist.append(box_dist)
                edges.append([[i, j], [j, i]])

                all_feats += [abs(box_dist), abs(box_dist)]
                all_edges += [[i, j], [j, i]]

        if len(dist) > 0:
            all_min_dist += [min(dist)]
            all_min_edges += edges[dist.index(min(dist))]

    ''''''
    max_dist = 1314.16032512918  # max(all_feats)
    all_min_dist = [x / max_dist for x in all_min_dist]
    all_feats = [x / max_dist for x in all_feats]
    if check_berry_occlusion:
        for b in all_strawberries:
            b['xmin'] = b['xmin'] / max_dist
            b['ymin'] = b['ymin'] / max_dist

    all_dist = {
        'dist': all_feats,
        'min_dist': all_min_dist,
        'edges': all_edges,
        'min_edges': all_min_edges
    }
    return all_dist


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