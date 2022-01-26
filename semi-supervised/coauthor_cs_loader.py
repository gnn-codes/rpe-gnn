import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random

class CoauthorCS():

    def __init__(self):

        dataset = Coauthor(root='data/Planetoid', name='CS', transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = "CoauthorCS"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        # self.train_label_rate = (int(self.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.data_contains_self_loops = data.contains_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

        self.train_mask, self.val_mask, self.test_mask = self.mask_generation()

    # adjacency list generation
    def adj_list_generation(self, edge_index):

        adj_list = [[] for n in range(self.num_nodes)]
        src_list = edge_index[0]
        dest_list = edge_index[1]

        for n in range(self.num_edges):

            adj_list[int(src_list[n])].append(int(dest_list[n]))

        return adj_list

    def data_visualize(self, img_name):

        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(img_name + ".png")
        plt.clf()


    def mask_generation(self):

        class_idx = [[] for i in range(self.num_classes)]
        train_mask = []
        val_mask = []
        test_mask = []
        for n in range(self.num_nodes):
            
            class_idx[self.node_labels[n]].append(n)

        # z = [len(class_idx[i]) for i in range(len(class_idx))]
        # print(z)
        
        for c in range(self.num_classes):

            # print("length ", len(class_idx[c]))
            sampled_c = random.sample(class_idx[c], 190 if len(class_idx[c]) >= 190 else len(class_idx[c])) 
            random.shuffle(sampled_c)
            train_set = sampled_c[:40]
            val_set = sampled_c[40:]
            train_mask += train_set
            val_mask += val_set
            test_set = list(set(class_idx[c]) - set(sampled_c))
            test_mask += test_set

        return train_mask, val_mask, test_mask

# coauthor_cs = CoauthorCS()
# print(coauthor_cs.num_nodes)
# print(coauthor_cs.num_edges)
# print(coauthor_cs.num_features)
# print(coauthor_cs.num_classes)

# print(len(coauthor_cs.train_mask), "  ", len(coauthor_cs.val_mask), "   ", len(coauthor_cs.test_mask))