import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random

class Cora():

    def __init__(self):

        dataset = Planetoid(root='data/Planetoid', name='Cora', transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = "Cora"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.train_label_rate = (int(data.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.data_contains_self_loops = data.contains_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

    # adjacency list generation
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


    def data_visualize(self, img_name):

        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(img_name + ".png")
        plt.clf()


# cora = Cora()
# print(type(cora.train_mask))
# test_idx = [i for i in range(cora.num_nodes) if cora.test_mask[i].item() is True]
# test_node = random.sample(test_idx, 1)
# print(test_node)
# print(cora.num_features)
# print(cora.train_mask)
# print(cora.edge_index)
# print(cora.train_mask.sum(), " ", cora.val_mask.sum(), " ", cora.test_mask.sum())
# print(cora.data)
# print(cora.edge_index[0][:30])
# print(cora.edge_index[1][:30])
# adj_list = cora.adj_list_generation(cora.edge_index)
# print(adj_list)
# path_list = cora.path_generator(adj_list, 64, 4)


# for n in range(cora.num_nodes):

    # print(n, "    ", adj_list[n])

# print("---------------------")

# for n in range(path_list.shape[0]):

#     print(path_list[n])

# print(path_list.shape)

# for e in range(cora.num_edges):

#     print(cora.edge_index[0][e], "   ", cora.edge_index[1][e])

# nodes = cora.num_nodes
# edges = cora.num_edges
# node_features = cora.node_features

# adj = torch.zeros(nodes, nodes)

# for e in range(edges):

    # src = cora.edge_index[0][e]
    # tgt = cora.edge_index[1][e]

    # print(cora.edge_index[0][e], "   ", cora.edge_index[1][e])

