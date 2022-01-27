import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
from coauthor_cs_loader import CoauthorCS

from sklearn.manifold import TSNE
import math
import os
import torch

from utils import UtilFunctions
from test import ModelEvaluation


class ModelTraining():

    def __init__(self):

        return

    def train(self, model, data_obj, nodes_batch, adj_list, train_iter, test_iter, opti, hidden_layers, l_range, n_sample, device):

        
        # model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + str(hidden_layers) + "_layers_.pt"
        model_path = None

        # training loop
        for epoch in range(train_iter):

            model.train()
            opti.zero_grad()
            emb, pred = model(nodes_batch, adj_list, l_range, n_sample)
            label = data_obj.node_labels
            pred = pred[data_obj.train_mask]
            label = label[data_obj.train_mask]
            pred = pred.to(device)
            label = label.to(device)
            loss = UtilFunctions.loss_fn(pred, label)
            loss.backward()
            opti.step()

            acc = ModelEvaluation().test(model, data_obj, nodes_batch, adj_list, test_iter, hidden_layers, model_path, l_range, n_sample, device, is_validation = True)
            print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}")

            model_path = os.getcwd() + "/saved_models/temp_models/" + data_obj.name.lower() + "_" + str(hidden_layers) + "_layers_" + str(epoch+1) + "_.pt"
            torch.save(model.state_dict(), model_path)

        return model_path
