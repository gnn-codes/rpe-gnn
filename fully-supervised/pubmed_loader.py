import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random

class PubMed():

    def __init__(self):

        dataset = Planetoid(root='data/Planetoid', name='PubMed', split = 'full', transform = NormalizeFeatures())
        data = dataset[0]

        self.name = "Pubmed"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        # self.train_mask = data.train_mask
        # self.val_mask = data.val_mask
        # self.test_mask = data.test_mask
        # self.train_mask_sum = data.train_mask.sum()
        # self.train_label_rate = (int(data.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.data_contains_self_loops = data.contains_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

        self.train_mask, self.val_mask, self.test_mask = self.mask_generation()

    def adj_list_generation(self, edge_index):

        adj_list = [[] for n in range(self.num_nodes)]
        src_list = edge_index[0]
        dest_list = edge_index[1]

        for n in range(self.num_edges):

            adj_list[int(src_list[n])].append(int(dest_list[n]))

        return adj_list

    # path generation
    def path_generator(self, adj_list, length, path_per_node):

        path_list = torch.zeros(path_per_node * self.num_nodes, length+1, dtype = torch.long)
        path_count = 0

        for n_idx in range(self.num_nodes):

            head_node = n_idx
            degree = len(adj_list[head_node])
            neighs = random.sample(adj_list[head_node], path_per_node if degree > path_per_node else degree)

            for _ in range(len(neighs)):

                path_list[path_count][0] = head_node
                curr_node = head_node

                for l in range(1, length+1):

                    next_node = random.sample(adj_list[curr_node], 1)[0]
                    path_list[path_count][l] = next_node
                    curr_node = next_node

                path_count += 1

        print("Path count ", path_count)
        path_list = path_list[:path_count]

        return path_list

    def mask_generation(self):

        class_idx = [[] for i in range(self.num_classes)]
        train_mask = []
        val_mask = []
        test_mask = []

        # storing classwise samples
        for n in range(self.num_nodes):

            class_idx[self.node_labels[n]].append(n)

        for c in range(self.num_classes):

            size = len(class_idx[c])
            train_size = int(size * 0.60)
            val_size = int(size * 0.20)
            test_size = size - train_size - val_size

            train_set = random.sample(class_idx[c], train_size)
            samples_without_train = list(set(class_idx[c]) - set(train_set))
            val_set = random.sample(samples_without_train, val_size)
            test_set = list(set(samples_without_train) - set(val_set))

            train_mask += train_set
            val_mask += val_set
            test_mask += test_set
            

        return train_mask, val_mask, test_mask 

    def data_visualize(self, img_name):

        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(img_name + ".png")
        plt.clf()


# pubmed = PubMed()
# print(len(pubmed.train_mask), "   ", len(pubmed.val_mask), "   ", len(pubmed.test_mask))
# print(int(pubmed.train_mask.sum()))
# print(pubmed.num_nodes)
# print(pubmed.num_edges)
# print(pubmed.contains_isolated_nodes)
# # print(cora.edge_index[0][:30])
# # print(cora.edge_index[1][:30])
# adj_list = pubmed.adj_list_generation(pubmed.edge_index)
# # print(adj_list)
# path_list = pubmed.path_generator(adj_list, 64, 5)


# for n in range(10):

#     print(adj_list[n])

# print("---------------------")

# for n in range(path_list.shape[0]):

    # print(path_list[n])






