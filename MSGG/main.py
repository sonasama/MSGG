import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LGCN import LGCN
import numpy as np
from utils.dataLoader import dataloader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
from sklearn import metrics
from torch.nn.functional import normalize
import scipy.io as sio

def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)

def adj_norm(adj, device):
    # 添加自环
    I = torch.eye(adj.size(0)).to(device)
    adj = adj + I

    # 计算度矩阵D
    D = adj.sum(1)

    # 计算D^(-1/2)
    D_inv_sqrt = torch.pow(D, -0.5).diag_embed()

    # 计算规范化邻接矩阵
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt

    # 确保对角线元素为1
    adj_normalized.fill_diagonal_(1)
    return adj_normalized

def train(args):
    features, gnd, p_labeled, p_unlabeled, adjs, adj_hats = dataloader(args.dataset_name, args.k, args.ratio)

    num_instance = adjs.shape[1]
    num_class = np.unique(gnd).shape[0]#GND为标记空间
    #num_instance = np.unique(gnd).shape[1]
    view_num = len(adjs)
    X = []
    for i in range(view_num):
        X.append(torch.from_numpy(features[0][i]).float().to(args.device))
    # X = torch.from_numpy(np.array(X))
    # print(type(X))
    # exit(0)

    N = gnd.shape[0] 

    gnd = torch.from_numpy(gnd).long().to(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    adj_hats = adj_hats.float().to(args.device)
    GCN_model = LGCN(N, X, view_num, args.dim_FCnet, args.dim_GCN, num_class).to(args.device)
    optimizer_GCN = torch.optim.Adam(GCN_model.parameters(),lr = 0.01) 

    cross_entropy_loss = nn.CrossEntropyLoss()

    ## Adj sampling
    for i in range(view_num):
        if i == 0:
            adj_S = adjs[i]
        else:
            adj_S = adj_S + adjs[i]
    adj_S = adj_S / view_num
    adj_S = adj_S / num_instance
    adj_S = adj_S.float().to(args.device)
    # normalization
    adj_S_normalized = adj_norm(adj_S, args.device)


    with tqdm(total=args.epoch_num, desc="Training") as pbar:
        for i in range(args.epoch_num):
            # GCN Training
            y_pred = GCN_model(X,adj_S_normalized)
            loss_GCN = cross_entropy_loss(y_pred[p_labeled], gnd[p_labeled])
            optimizer_GCN.zero_grad()
            loss_GCN.backward()
            optimizer_GCN.step()

            pbar.set_postfix({
                            'Loss_GCN' : '{0:1.5f}'.format(loss_GCN)})
            pbar.update(1)
    pred_label_for_tsne = F.log_softmax(y_pred,1)
    pred_label = torch.argmax(F.log_softmax(y_pred,1), 1)
    accuracy_value = accuracy_score(gnd[p_unlabeled].cpu().detach().numpy(), pred_label[p_unlabeled].cpu().detach().numpy())
    F1 = metrics.f1_score(gnd[p_unlabeled].cpu().detach().numpy(), pred_label[p_unlabeled].cpu().detach().numpy(), average='macro')
    sio.savemat("Y3S.mat", {'labe':gnd[p_unlabeled].cpu().detach().numpy()})
    sio.savemat("resp3S.mat", {'resp': pred_label_for_tsne[p_unlabeled].cpu().detach().numpy()})
    print(args.dataset_name, "Accuracy:", accuracy_value)
    print(args.dataset_name, "F1:", F1)
    return accuracy_value, F1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run .")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--dataset-name", nargs = "?", default = "3source")
    parser.add_argument("--epoch-num", type = int, default = 500, help = "Number of training epochs.")
    parser.add_argument("--seed", type = int, default = 2023, help = "Random seed for network training.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate. Default is 0.01.")

    parser.add_argument("--ratio", type = float, default = 0.1, help = "Training ratio")
    #parser.add_argument("--epoch_num", type = float, default = 5, help = "Number of repeated times")
    parser.add_argument("--k", type = int, default = 15 , help = "k of KNN graph.")

    parser.add_argument("--dim-FCnet", type = int, default = 128, help = "dim_FCnet")
    parser.add_argument("--dim-GCN", type = int, default = 64, help = "dim_GCN")

    args = parser.parse_args()


    print(args.dataset_name)
    accu = np.zeros((2,5))
    F3 = np.zeros((1,5))
    for i in range(1):#跑几轮的地方
        accuracy_value, F3 = train(args)
        accu[0, i] = accuracy_value
        accu[1, i] = F3

    mean_accu = np.mean(accu[0])

    # 计算方差
    var_accu = np.var(accu[0])
    mean_F1 = np.mean(accu[1])

    # 计算方差
    var_F1 = np.var(accu[1])
    print("Mean Accuracy:", mean_accu)
    print("Accuracy Variance:", var_accu)
    print("Mean F1:", mean_F1 )
    print("Accuracy F1:", var_F1)