import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCL import GraphConvolution
from model.FC_net import FCnet

class LGCN(nn.Module):
    def __init__(self, N, X, view_num, dim_FCnet, dim_GCN, num_class):
        super(LGCN,self).__init__()


        self.view_num = view_num
        
        self.FCnet_model = nn.ModuleList()
        for i in range(self.view_num):
            self.FCnet_model.append(FCnet(N, X[i].size(1), [X[i].size(1) // 2, dim_FCnet]))

    
        self.gc1 = GraphConvolution(dim_FCnet * self.view_num, dim_GCN)
        self.gc2 = GraphConvolution(dim_GCN, num_class)



    def forward(self, X, adj):
        H_FC = []
        for i in range(self.view_num):
            H_FC.append(self.FCnet_model[i](X[i]))

        H_merge = torch.cat(H_FC, dim=1)

        Z = torch.relu(self.gc1(H_merge, adj))  
        Z = F.dropout(Z, p=0.3)
        Z = self.gc2(Z, adj)

        return Z
    
