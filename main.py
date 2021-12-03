#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:58:38 2021

@author: illusionist
"""

# main.py

import torch
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve

from util import load_data, EarlyStopping
from model_hetero import Model, HAN, HeteroDotProductPredictor
import numpy as np
import csv
from csv import DictReader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
import dgl
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('always')

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def compute_score(pos_score, neg_score):
    sortedList=pos_score.tolist()
    sortedList=sortedList.sort()
    print(sortedList)
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    return roc_auc_score(labels, scores), f1_score(labels, scores.round(), average='micro'), f1_score(labels, scores.round(), average='macro'), recall_score(labels, scores.round(), average='micro'), precision_score(labels, scores.round(), average='micro'), average_precision_score(labels, scores.round(), average='micro')
    #

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #args['device'] = device
    g = load_data()
    drug_feats = g.nodes['drug'].data['hv']
    protein_feats = g.nodes['protein'].data['hv']
    chem_Struct_feats = g.nodes['chem_Struct'].data['hv']
    side_effect_feats = g.nodes['side_effect'].data['hv']
    node_features = {'drug': drug_feats, 'protein': protein_feats, 'chem_Struct': chem_Struct_feats, 'side_effect': side_effect_feats}
    #node_features = g.ndata['hv']
    n_features = drug_feats.shape[1]
    model = Model(#graph=g,
                    #features=features,  
                    meta_paths=[['dp', 'pd'], ['dp','pp', 'pd'], ['ds', 'sd'] , ['dcs', 'csd']],
                    in_size=n_features,
                    hidden_size=args['hidden_units'],
                    out_size=2,
                    num_heads=args['num_heads'],
                    dropout=args['dropout'])
    #model = Model(n_features, 16, 2, g.etypes)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(11):
        negative_graph = construct_negative_graph(g, 12, ('drug', 'ddi', 'drug'))
        pos_score, neg_score = model(g, negative_graph, 'drug', drug_feats, ('drug', 'ddi', 'drug'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        auc, micro_f1, macro_f1, recall, precision, aupr = compute_score(pos_score, neg_score)
        #auc, micro_f1, macro_f1 = compute_score(pos_score, neg_score)
        if epoch==10:
          print('AUROC: {} Micro-f1: {} Macro-f1: {} Recall: {} Precision: {} AUPR: {}'.format(auc, micro_f1, macro_f1, recall, precision, aupr))
          #print('AUROC: {} Micro-f1: {} Macro-f1: {}'.format(auc, micro_f1, macro_f1))

        
if __name__ == '__main__':
    import argparse

    from util import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
