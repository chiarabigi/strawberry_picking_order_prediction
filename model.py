import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, global_mean_pool, GCNConv
from torch_geometric.utils import softmax


class GCN_scheduling(torch.nn.Module):
    def __init__(self, hidden_layers, num_layers):
        super(GCN_scheduling, self).__init__()

        self.nfeat = 6  # number of node features
        self.nhead = 5
        self.edge_dim = 1  # size of edge feature

        self.conv1 = GATConv(self.nfeat, hidden_layers, heads=self.nhead, edge_dim=1)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GATConv(self.nhead*hidden_layers, hidden_layers, heads=self.nhead))

        self.conv2 = GATConv(self.nhead*hidden_layers, 1, heads=self.nhead, edge_dim=1, concat=False)

        self.linear = Linear(1, 1, bias=False, weight_initializer='glorot')

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = self.conv1(x, edge_index, edge_weight)  # new node features
        x2 = F.relu(x1)

        x3 = F.dropout(x2, p=0.2, training=self.training)
        x4 = self.conv2(x3, edge_index, edge_weight)

        x5 = self.linear(x4)
        x6 = self.sigmoid(x5)
        return x6
