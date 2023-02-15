import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear


class GCN_scheduling(torch.nn.Module):
    def __init__(self, hidden_layers, num_layers):
        super(GCN_scheduling, self).__init__()

        self.nfeat = 262  # number of node features
        self.nhead = 8
        self.edge_dim = 1  # size of edge feature

        self.conv1 = GATConv(self.nfeat, hidden_layers, heads=self.nhead)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GATConv(self.nhead*hidden_layers, hidden_layers, heads=self.nhead))

        self.conv2 = GATConv(self.nhead*hidden_layers, hidden_layers // 2, heads=self.nhead)

        self.lin1 = Linear(self.nhead*hidden_layers // 2, self.nhead, bias=False, weight_initializer='glorot')
        self.lin2 = Linear(self.nhead, 1, bias=False, weight_initializer='glorot')

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = self.conv1(x, edge_index)  # new node features
        x2 = F.elu(x1)

        for conv in self.convs:
            x2 = conv(x2, edge_index)
            x2 = F.elu(x2)

        x3 = F.dropout(x2, p=0.2, training=self.training)
        x4 = self.conv2(x3, edge_index)
        x5 = self.lin1(x4)

        x6 = F.elu(x5)
        x7 = self.lin2(x6)
        x8 = self.sigmoid(x7)
        return x8
