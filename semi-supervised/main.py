import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os 

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
from coauthor_cs_loader import CoauthorCS

from train import ModelTraining
from test import ModelEvaluation
from utils import UtilFunctions

from sklearn.manifold import TSNE
from model import RPE_GNN
import argparse

def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--lr', help = 'learning rate', default = 0.2, type = float)
    parser.add_argument('--hidden_layers', help = 'number of hidden layers', default = 2, type = int)
    parser.add_argument('--hidden_dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 100, type = int)
    parser.add_argument('--test_iter', help = 'number of test iterations', default = 1, type = int)
    parser.add_argument('--use_saved_model', help = 'use saved model in directory', default = False, type = None)
    parser.add_argument('--l_range', help = 'number of layers the branching is allowed', default = False, type = int)
    parser.add_argument('--n_sample', help = 'sampling neighbors per node', default = False, type = int)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser

parsed_args = argument_parser().parse_args()

dataset = parsed_args.dataset
lr = parsed_args.lr
hidden_layers = parsed_args.hidden_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
l_range = parsed_args.l_range
n_sample = parsed_args.n_sample
device = parsed_args.device
# print("Device Used: ", device)

if dataset == 'cora':

    data_obj = Cora()

elif dataset == 'citeseer':

    data_obj = CiteSeer()

elif dataset == 'pubmed':

    data_obj = PubMed()

elif dataset == 'coauthorcs':

    data_obj = CoauthorCS()

else:

    print("Incorrect name of dataset")


# adjacency list generation
adj_list = UtilFunctions().adj_list_generation(data_obj.edge_index, data_obj.num_nodes, data_obj.num_edges)
nodes_batch = [i for i in range(data_obj.num_nodes)]

model = RPE_GNN(data_obj, num_layers = hidden_layers, hidden_dim = hidden_dim, device = device)
opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)

# print("Model Name: RPE-GNN  -----  Semi-supervised Settings")
print("Dataset: ", dataset.upper())
# print("Hidden Layers:", hidden_layers)

if use_saved_model == 'False':

    # training of the model
    print("Optimization started....")
    trainer = ModelTraining()
    model_path = trainer.train(model, data_obj, nodes_batch, adj_list, train_iter, test_iter, opti, hidden_layers, l_range, n_sample, device)

else:

    # print("Trained model successfully loaded...")
    model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + str(hidden_layers) + "_layers_.pt"


# evaluation
# print("Evaluating on Test set")
avg_acc = 0.0
max_acc = 0.0
eval = ModelEvaluation()

for i in range(test_iter):

    acc = eval.test(model, data_obj, nodes_batch, adj_list, test_iter, hidden_layers, model_path, l_range, n_sample, device, is_validation = False)
    if acc > max_acc:
        max_acc = acc
    avg_acc += acc
    print("Test iteration:", i+1, " complete --- accuracy ",acc)

avg_acc /= test_iter

# print(f'Maximum accuracy on Test set: {max_acc:.4f}')
print(f'Average accuracy on Test set: {avg_acc:.4f}')




