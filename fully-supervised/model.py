import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
# from coauthor_cs_loader import CoauthorCS
from sklearn.manifold import TSNE
import math


class RPE_NET(nn.Module):
  
    def __init__(self, input_dim, output_dim): 

        super(RPE_NET, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.path_weight_1 = nn.Linear(input_dim, output_dim)
        self.path_weight_2 = nn.Linear(input_dim, output_dim)


    def forward(self, nodes, features, adj_list, curr_neigh, curr_embed, curr_target, layer, l_range, n_sample):

        '''
        nodes --- batch of nodes whose embeddings will be estimated
        features --- set of features of the nodes in the graph
        adj_list --- adjacency list of the graph
        curr_neigh --- list of elemensts of current neighbors of each nodes in the batch
        curr_embed --- stroes current path embeddings for nodes in the batch upto kth hop neighborhood
        curr_target --- stores the current target nodes pairs
        layer --- number of current layer
        '''

        '''
        Cora --- 
                layer = 2  ===>  l_range = , n_sample = 
                layer = 4  ===>  l_range = , n_sample = 
                layer = 8  ===>  l_range = 3, n_sample = 3
                layer = 16 ===>  l_range = , n_sample = 
                layer = 32 ===>  l_range = , n_sample =  
                layer = 64 ===>  l_range = , n_sample = 

        Citeseer --- 
                layer = 2 ===>  l_range = , n_sample = 
                layer = 4 ===>  l_range = , n_sample = 
                layer = 8 ===>  l_range = 2, n_sample = 3
                layer = 16 ===> l_range = , n_sample = 
                layer = 32 ===> l_range = , n_sample = 
                layer = 64 ===> l_range = , n_sample = 

        Pubmed --- 
                layer = 2 ===>  l_range = , n_sample = 
                layer = 4 ===>  l_range = , n_sample = 
                layer = 8 ===>  l_range = 3, n_sample = 3
                layer = 16 ===> l_range = , n_sample = 
                layer = 32 ===> l_range = , n_sample = 
                layer = 64 ===> l_range = , n_sample =  
        '''

        path_id = []
        counter = 0
        neigh_list = []

        if layer <= l_range:
            num_neighs = n_sample
        else:
            num_neighs = 1 
        
        # union of all sampled neighbors of current parent nodes
        for n in curr_neigh:

            degree = len(adj_list[n])
            neighs = random.sample(adj_list[n], num_neighs if degree >= num_neighs else degree)
            neigh_list += neighs
            p_id = [counter for i in range(len(neighs))]
            path_id += p_id
            counter += 1

        # print(layer, "  ", len(neigh_list))
        parent_list = [curr_neigh[i] for i in path_id]
        new_target = [curr_target[i] for i in path_id]
        curr_feat = curr_embed[path_id]

        # updating current embedding from kth hop to (k+1)th hop
        parent_feat = features[parent_list]
        child_feat = features[neigh_list]
        feat_diff = torch.sub(child_feat, parent_feat)

        new_feat = self.path_weight_1(feat_diff)
        new_feat = F.relu(new_feat)
        curr_feat = self.path_weight_2(curr_feat)
        curr_feat = F.relu(curr_feat)

        new_embed = torch.add(curr_feat, new_feat)

        curr_neigh = neigh_list
        curr_embed = new_embed
        curr_target = new_target

        return curr_neigh, curr_embed, curr_target


# defining deep gnn models
class RPE_GNN(nn.Module):

    def __init__(self, data_obj, num_layers, hidden_dim, dropout, device):
        super(RPE_GNN, self).__init__() 

        self.num_layers = num_layers
        self.data_obj = data_obj
        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = dropout

        self.trans_feat = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.pred_trans = nn.Linear(self.data_obj.num_features, self.data_obj.num_classes)

        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        for l in range(self.num_layers):

            if l == self.num_layers - 1:

                self.gcn_convs.append(RPE_NET(self.hidden_dim, self.data_obj.num_classes).to(self.device))

            else:

                self.gcn_convs.append(RPE_NET(self.hidden_dim, self.hidden_dim).to(self.device))

            self.lns.append(nn.LayerNorm(self.hidden_dim).to(self.device))


    def forward(self, nodes_batch, adj_list, l_range, n_sample):

        features = self.trans_feat(self.data_obj.node_features).to(self.device)
        pred_features = self.pred_trans(self.data_obj.node_features).to(self.device)

        curr_neigh = nodes_batch     
        curr_embed = torch.zeros(len(nodes_batch), self.hidden_dim).to(self.device)
        curr_target = [n for n in nodes_batch]           
     
        # s = random.randint(1, 1e5)
        # print(s)
        # random.seed(s)

        # message propagation through the hidden layers (K-hop message propagation)
        for i in range(self.num_layers):

            curr_neigh, curr_embed, curr_target = self.gcn_convs[i](nodes_batch, features, adj_list, curr_neigh, curr_embed, curr_target, i, l_range, n_sample)

            if i != self.num_layers - 1:

                curr_embed = self.lns[i](curr_embed)
                curr_embed = F.dropout(curr_embed, p = self.dropout, training = self.training)
    

        # final node embedding estimation
        embedding = torch.zeros(len(nodes_batch), self.data_obj.num_classes).to(self.device)
        path_count = [0 for i in range(len(nodes_batch))]
        node_id_map = {n:i for i,n in enumerate(nodes_batch)}

        # final aggregation step
        for i in range(len(curr_embed)):

            f = (curr_embed[i].squeeze(0) * pred_features[curr_neigh[i]]).to(self.device)
            norm = torch.norm(f, p = 2)
            if norm != 0 :
                f = f / norm
            node_id = node_id_map[curr_target[i]]
            path_count[node_id] += 1
            embedding[node_id] += f

        path_count = [1 if path_count[i] == 0 else path_count[i] for i in range(len(path_count))]
        path_count = torch.tensor(path_count)
        path_count = path_count.to(self.device)
        embedding = embedding.div(path_count.unsqueeze(1))
        embedding = torch.add(embedding, pred_features)

        x = embedding
        x = F.log_softmax(x, dim = 1)

        return embedding, x