import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, GCNConv
from customLeaky import CustomLeakyReLU


class GCN_scheduling(torch.nn.Module):
    def __init__(self, hidden_layers, num_layers):
        super(GCN_scheduling, self).__init__()

        self.nfeat = 4  # number of node features
        self.nhead = 1
        self.edge_dim = 1  # size of edge feature

        self.conv1 = GATConv(self.nfeat, hidden_layers)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GCNConv(self.nhead*hidden_layers, hidden_layers))

        self.conv2 = GCNConv(self.nhead*hidden_layers, 1)

        self.linear = Linear(1, 1, bias=False, weight_initializer='glorot')

        self.sigmoid = torch.nn.Sigmoid()
        self.customSigmoid = mySigmoid(2)
        self.customleaky = CustomLeakyReLU()
        self.leaky = torch.nn.LeakyReLU(0.01)
        self.prelu = torch.nn.PReLU()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, data):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = self.conv1(x, edge_index, edge_weight)  # new node features
        x2 = self.prelu(x1)

        x3 = F.dropout(x2, p=0.2, training=self.training)
        x4 = self.conv2(x3, edge_index, edge_weight)
        #x5 = self.linear(x4)
        x5 = self.sigmoid(x4)
        #x5 = self.customleaky(x4)
        #x5 = self.prelu(x4)
        #x5 = self.lin(x4)
        return x5


class mySigmoid(torch.nn.Module):
    def __init__(self, beta):
        super(mySigmoid, self).__init__()
        self.beta = beta
    def forward(self, data):
        x = data - self.beta
        output = 1 / (1 + torch.exp(-x))
        return output
