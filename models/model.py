import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from kinematics import ForwardKinematicsURDF


class SpatialBasicBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='add', batch_norm=False, bias=True, **kwargs):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out
        out += self.upsample(x[1])
        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.leaky_relu(self.lin(z))


class Encoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Encoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=channels, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=32, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=32, out_channels=64, edge_channels=dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Decoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=64+2, out_channels=32, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=32, out_channels=16, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=16, out_channels=channels, edge_channels=dim)

    def forward(self, x, edge_index, edge_attr, lower, upper):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        x = torch.cat([x, lower, upper], dim=1)
        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr).tanh()
        return out


class YumiNet(torch.nn.Module):
    def __init__(self):
        super(YumiNet, self).__init__()
        self.encoder = Encoder(6, 3)
        self.transform = nn.Sequential(
            nn.Linear(6*64, 14*64),
            nn.Tanh(),
        )
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsURDF()
    
    def forward(self, data, target):
        return self.decode(self.encode(data), target)
    
    def encode(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        z = self.transform(z.view(data.num_graphs, -1, 64).view(data.num_graphs, -1)).view(data.num_graphs, -1, 64).view(-1, 64)
        return z
    
    def decode(self, z, target):
        ang = self.decoder(z, target.edge_index, target.edge_attr, target.lower, target.upper)
        ang = target.lower + (target.upper - target.lower)*(ang + 1)/2
        pos, rot, global_pos = self.fk(ang, target.parent, target.offset, target.num_graphs)
        return ang, pos, z, None, None, rot, global_pos
    
    def forward_kinematics(self, ang, target):
        pos, rot, global_pos = self.fk(ang, target.parent, target.offset, target.num_graphs)
        return ang, pos, None, None, None, rot, global_pos
