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
import torch
import os
import torch.nn.functional as F

class UtilFunctions():

    def __init__(self):

        return

    # visuals of embedding
    def visualize(self, feat_map, color, name, hidden_layers):
        z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(os.getcwd() + "/visuals/" + name + "_" + str(hidden_layers) + "_layers_embedding.png")
        plt.clf()
        plt.close()

    # loss function
    def loss_fn(pred, label):

        # return F.nll_loss(pred, label)
        return F.cross_entropy(pred, label)

    def adj_list_generation(self, edge_index, num_nodes, num_edges):

        adj_list = [[] for n in range(num_nodes)]

        for n in range(num_edges):

            src = int(edge_index[0][n])
            neigh = int(edge_index[1][n])
            adj_list[src].append(neigh)

        return adj_list
