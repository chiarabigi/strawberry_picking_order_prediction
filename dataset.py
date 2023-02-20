import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from scipy.spatial import distance
import json
from utils import only_sides, distances
import copy

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class SchedulingDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def anns(self):
        with open(self.raw_paths[0]) as f:
            anns = json.load(f)  # json file with the annotations of bbox occlusion
        return anns

    @property
    def raw_file_names(self):
        return 'gnnann.json'

    @property
    def processed_file_names(self):  # don't process if you find this file
        return 'data_0.pt'  # 'not_yet.pt'

    def download(self):
        pass

    def process(self):
        idx = 0
        anns=self.anns
        for index in range(len(anns)):
            # Read data from `raw_path`.
            box_obj = anns[index]['boxes']
            coord = anns[index]['img_ann']
            sched = anns[index]['sc_ann']
            occ = anns[index]['occ_ann']
            occ_score = anns[index]['occ_score']
            occ_leaf = anns[index]['occ_leaf']
            easiness = anns[index]['easiness']
            patches = anns[index]['patches']
            ripeness = anns[index]['ripeness']
            students_scheduling = anns[index]['students_sc_ann']
            heuristic_scheduling = anns[index]['heuristic_sc_ann']
            # Get node features
            node_feats = self._get_node_features(coord, occ_score, ripeness, occ_leaf, patches)
            # Get edge features and adjacency info
            edge_feats, edge_index = self.knn(box_obj)

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=torch.tensor(easiness, dtype=torch.float32, device=device).unsqueeze(1),
                        # scheduling=scheduling,  # 1 0 classes
                        students_ann=torch.tensor(students_scheduling, dtype=torch.int32, device=device).unsqueeze(1),
                        heuristic_ann=torch.tensor(heuristic_scheduling, dtype=torch.int32, device=device).unsqueeze(1),
                        label=torch.tensor(sched, dtype=torch.int32, device=device).unsqueeze(1),  # gt
                        info=torch.tensor(occ, device=device).unsqueeze(1)
                        )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1


    def _get_node_features(self, box, occ, ripenes, occ_leaf, patches):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for a in range(len(box)):
            ''''''
            if len(patches) > 0:
                patchesa = patches[a]
            else:
                patchesa = patches
            node_feats = []
            # Feature 0: x
            node_feats.append(box[a][0])
            # Feature 1: y
            node_feats.append(box[a][1])
            # Feature 2: Width
            # node_feats.append(box[a][2])
            # Feature 3: Height
            # node_feats.append(box[a][3])
            # Feature 4: Ripeness
            node_feats.append(ripenes[a])
            # Feature 5: Occlusion berry %
            node_feats.append(occ[a])  # the occlusion score was computed in the 'easiest' script
            # Feature 6: Occlusion leaf
            node_feats.append(occ_leaf[a])
            # Feature 6-86: Image patch
            node_feats.extend(patchesa)

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)

        return torch.tensor(all_node_feats, dtype=torch.float32, device=device)

    def knn(self, box):  # "k-nearest neighbour"
        # Compute all possible edges (sides + diagonals), ONCE
        edge_feats, edge_indices, min_dist, min_edges = distances(box)

        sides_feats, sides_indices = only_sides(edge_feats, edge_indices, box)
        # Take only diagonals
        if len(box) > 3:
            diag_feats = [ele for ele in edge_feats if ele not in sides_feats]
            diag_indices = [ele for ele in edge_indices if ele not in sides_indices]
            edge_feats = diag_feats
            edge_indices = diag_indices
        min_dist = [round(x, 7) for x in min_dist]
        edge_feats = [round(x, 7) for x in edge_feats]
        edge_feats.extend([x for x in min_dist if x not in edge_feats])
        edge_indices.extend([x for x in min_edges if x not in edge_indices])

        edge_indices = torch.tensor(edge_indices, dtype=torch.float32, device=device)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32, device=device)
        return edge_feats, edge_indices


    def _get_scheduling(self, label):
        for i in range(len(label)):
            '''
            if label[i] == len(label):
                label[i] = 0
            elif label[i] == 2:
                label[i] = 0.95
            elif label[i] == 3:
                label[i] = 0.5
            elif len(label) > 3:
                label[i] = (len(label) - label[i]) / (2*(len(label) - 3))'''

            '''if label[i] == 2:
                label[i] = 1  # 0.91
            elif label[i] > 2 or label[i] == -1:
                label[i] = 0'''
            if label[i] != 1:  # weights in BCEloss
                label[i] = 0

        return torch.tensor(label, dtype=torch.float32, device=device).unsqueeze(1)

    def len(self):
        return len(self.anns)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), map_location=device)
        return data
