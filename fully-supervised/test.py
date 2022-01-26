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
from utils import UtilFunctions


class ModelEvaluation():

    def __init__(self):

        return

    def test(self, model, data_obj, nodes_batch, adj_list, test_iter, hidden_layers, model_path, l_range, n_sample, device, is_validation):

        if is_validation is False:

            model.load_state_dict(torch.load(model_path))

        model.eval()
        correct = 0
        emb, pred = model(nodes_batch, adj_list, l_range, n_sample)
        pred = pred.argmax(dim = 1)
        label = data_obj.node_labels
        pred = pred[data_obj.test_mask]
        label = label[data_obj.test_mask]
        pred = pred.to(device)
        label = label.to(device)
        correct = pred.eq(label).sum().item()
        # accuracy = correct / int(data_obj.test_mask.sum())
        accuracy = correct / len(data_obj.test_mask)

        if is_validation is False:

            UtilFunctions().visualize(emb, data_obj.node_labels, data_obj.name, hidden_layers)

        return accuracy
