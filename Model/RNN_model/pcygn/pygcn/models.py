import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        assert len(x.shape)== 4
        B, L, H, W = x.shape
        inputs = x.reshape(B*L, H, W)
        outputs = []
        for input in inputs:
            input = F.relu(self.gc1(input, adj))
            input = F.dropout(input, self.dropout, training=self.training)
            input = self.gc2(input, adj)
            outputs.append(input.unsqueeze(0))
        out = torch.cat(outputs, dim=0)
        out = out.reshape(B,L,H,W)
        return F.log_softmax(out, dim=-1)

        
        
        #     x = F.relu(self.gc1(x, adj))
        #     x = F.dropout(x, self.dropout, training=self.training)
        #     x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
