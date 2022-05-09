import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Pygcn.pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nin, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
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
 
 
class RNN(nn.Module):
   def __init__(self, rnn_in_features, rnn_hidden_features, rnn_out_features):
    super(RNN, self).__init__()
    self.rnn_in_features = rnn_in_features
    self.rnn_hidden_features = rnn_hidden_features
    self.rnn_out_features = rnn_out_features
    self.model = nn.RNN(input_size = self.rnn_in_features, hidden_size = self.rnn_hidden_features, batch_first=True)
    self.out =  nn.Linear(self.rnn_hidden_features, self.rnn_out_features)
    
   def forward(self, x): # input = [B, L=12, H, W]
      B, L, H, W = x.shape
      h0 = torch.zeros(1, B, self.rnn_hidden_features).to(x.device) # here, 1 indicates one direction
    #   print("h0.shape: ", h0.shape)
      self.model.flatten_parameters()
      res, hn = self.model(x.reshape(B, L , H*W))
    #   print("after rnn, res.shape, hn.shape is: ", res.shape, hn.shape) # after rnn, res.shape, hn.shape is:  torch.Size([32, 12, 36]) torch.Size([1, 32, 36])
      out = F.relu(self.out(res))
    #   print("out.shape: ", out.shape) # out.shape:  torch.Size([32, 12, 6])
      return out
 
 
class CityBike_Model(nn.Module):
 def __init__(self, device, 
              gcn_in_dim, gcn_hid_dim, gcn_out_dim, gcn_dropout,
              rnn_in_features, rnn_hidden_features, rnn_out_features, 
              final_outfeatures =6):
     super().__init__()
     self.GCN = GCN(nin=gcn_in_dim, nhid = gcn_hid_dim, nout = gcn_out_dim, dropout=gcn_dropout).to(device)
     self.rnn = RNN(rnn_in_features=rnn_in_features, rnn_hidden_features=rnn_hidden_features, rnn_out_features = rnn_out_features).to(device)
     self.mlp = nn.Linear(rnn_out_features, final_outfeatures).to(device)
 def forward(self, x, adj = torch.randn(6,6)):
     adj = adj.to(x.device)
     out =  self.GCN(x, adj)
    #  print("after GCN: out.shape: ", out.shape) # after GCN: out.shape:  torch.Size([32, 11, 6, 6])
    #  self.rnn.flatten_parameters()
     out = self.rnn(out)
    #  print("after RNN: out.shape: ", out.shape) # after RNN: out.shape:  torch.Size([32, 11, 6])
    #  print("out.dtype: ", out.dtype) # out.dtype:  torch.float32 ->   torch.float64
     out = self.mlp(out)#.double()
     return out
   


if __name__ == "__main__":
  ## test GNN
  input = torch.randn(32, 12, 6, 6)
  model0 = GCN(nin=6, nhid = 6, nout = 6, dropout=0.1)
  adj = torch.randn(6,6)
  out =  model0(input, adj)
  print("after GCN: out.shape: ", out.shape) # after GCN: out.shape:  torch.Size([32, 12, 6, 6])

  
  ##: test RNN
  #  input = torch.randn(32, 12, 36)
  model = RNN(rnn_in_features=36, rnn_hidden_features=36, rnn_out_features = 6)
  out = model(out)
  print("after RNN: out.shape: ", out.shape) # after RNN: out.shape:  torch.Size([32, 12, 6])


