import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import json
from utils.edges import only_sides, distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PickingSuccessDataset(Dataset):
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
            # coord = anns[index]['img_ann']
            # min_dist = anns[index]['min_dist']
            patches = anns[index]['patches']
            ripeness = anns[index]['ripeness']

            # occ = anns[index]['occ_ann']
            occ_score = anns[index]['occ_score']
            occ_leaf = anns[index]['occ_leaf']
            target = anns[index]['target_vector']

            y = torch.tensor(anns[index]['success'], dtype=torch.float)

            # Get node features
            node_feats = self._get_node_features(box_obj, occ_score, occ_leaf, patches, target)
            # Get edge features and adjacency info
            edge_feats, edge_index = self.knn(box_obj)
            # Get occlusion info
            # info = self._get_occlusion(occ)

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=y
                        )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1


    def _get_node_features(self, box, occ, occ_leaf, patches, target):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for a in range(len(occ)):
            ''''''
            if len(patches) > 0:
                patch = patches[a]
            else:
                patch = patches
            node_feats = [box[a][0], box[a][1], occ[a], occ_leaf[a]]
            # Feature 0: x min
            # Feature 1: y min
            # Feature 2: Width
            # Feature 3: Height
            # Feature 5: Occlusion berry %
            # Feature 6: Occlusion leaf 0/1
            # Feature 7-1280something: Image patch
            if target[a]:
                node_feats.append(1)
            else:
                node_feats.append(0)
            node_feats.extend(patch)

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
        # add the edges of the minimum distances, if not already present
        min_dist = [round(x, 7) for x in min_dist]
        edge_feats = [round(x, 7) for x in edge_feats]
        add_edge = [x for x in min_edges if x not in edge_indices]
        add_idx = [min_edges.index(x) for x in add_edge]
        add_feats = [min_dist[x] for x in add_idx]
        edge_feats.extend(add_feats)
        edge_indices.extend(add_edge)

        edge_indices = torch.tensor(edge_indices, dtype=torch.float32, device=device)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32, device=device)
        return edge_feats, edge_indices


    def _get_occlusion(self, occ):
        occlusion_properties = ['occluded', 'occluding', 'occluded/occluding', 'neither']
        info=[]
        for i in range(len(occ)):
            info.append(occlusion_properties[occ[i]])
        return info

    def len(self):
        return len(self.anns)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
