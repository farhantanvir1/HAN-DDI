#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:49:05 2021

@author: illusionist
"""

# util.py

import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy.sparse as sp
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

from csv import DictReader
from csv import reader
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
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 200,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['hetero']=True
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = 100000000000
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

"""
    this function creates a dictionary for entity(drug/protein/species/indication) sent as argument
"""

def create_dictionary(path):
    
    count=1
    dict_val={}
    # dictionary's key will content entity object's id(drug id/protein id)
    # value will contain the index id where corresponding entity object's data will be found in adjacency and
    # commuting matrix
    with open(path) as f:
        content = f.readlines()
        for x in content:
            dict_val[x.strip()]=count
            count+=1
    
    
    return dict_val


"""
    this function creates adjacency matrix for 2 entities' dictionaries sent as arguments
    it has 7 parameters. 
    first parameter refers to file path
    
    second and third parameter refers to first entity's desired column name in file(where 
    we will look for values of first entity) and first entity's dictonary
    
    forth and fifth parameter refers to second entity's desired column name in file(where 
    we will look for values of second entity) and second entity's dictonary
    
    sixth and seventh parameter indicates whether first and second colummn values have multiple values.
    in case of multiple values, values maybe split by semicolons.
    if it is, we have to do some text processing.
    
    sixth parameter indicates whether the first entity is drug.
    drug data maybe split by semicolons.
    if it is, we have to do some text processing on drug ids. only then, we can obtain accurate index id 
    from drug's dictionary.
    
    seventh parameter indicates whether the meta-path, adjacency matrix is desired for, 
    is of ADE/Indication relationship or not
    
    if the relationship type is ADE/Indication, then the eighth parameter refers to the type of relationship
    
"""

def create_adjacency_matrix(path, col1, dict1, col2, dict2, FirstColHasMultiValue, SecondColHasMultiValue):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
 
        
        csv_dict_reader = DictReader(read_obj)
        
        next(csv_dict_reader, None)

        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:

            # considering different combinations when columns maybe semicolon-seperated or not
            # for each case, null checking is done
            if FirstColHasMultiValue==False:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        adj_mat[dict1[row[col1]]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        items=row[col2].split('; ')
                        for item in items:
                            adj_mat[dict1[row[col1]]][dict2[item]]=1
            elif FirstColHasMultiValue==True:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        items=row[col1].split('; ')  
                        adj_mat[dict1[item]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        items1=row[col1].split('; ')
                        items2=row[col2].split('; ')
                        for item1 in items1:
                            for item2 in items2:
                                adj_mat[dict1[item1]][dict2[item2]]=1
            """
            elif isDrug1==False and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items=row[col2].split('; ')
                    for item in items:
                        adj_mat[dict1[row[col1]]][dict2[item]]=1
             
            elif isDrug1==True and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items_1=row[col1].split('; ')
                    items_2=row[col2].split('; ')
                    for item_1 in items_1:
                         for item_2 in items_2:
                             adj_mat[dict1[item_1]][dict2[item_2]]=1
            """

    return adj_mat

def create_adjacency_matrix_drug_se(path, col1, dict1, col2, dict2, FirstColHasMultiValue, SecondColHasMultiValue):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    df = pd.read_excel(path)
    for index, row in df.iterrows():
      if row[col1] in dict1.values() and row[col2] in dict2.values():
        adj_mat[dict1[row[col1]]][dict2[row[col2]]]=1
      #print(row['c1'], row['c2'])
    
    return adj_mat

def create_adjacency_matrix_chemical_structure(path, col1, col2, dict1, num):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, 513)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            if row[col2]!='':
                for i in range(num):
                    adj_mat[dict1[row[col1]]][i]=int(row[col2][i+1])

    return adj_mat

def create_adjacency_matrix_labelled_data(path, col1, col2, col3, dict1, dict2):
    
    adj_mat=[[]]
    
    #initializing all values in adjacency matrix to 0
    #getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict1.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #rows = []
    #cols = ['DrugBank 1', 'DrugBank 2','Polypharmacy Side Effect',	'Side Effect Name'] 
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        csv_dict_reader = DictReader(read_obj)
        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:
            # row variable is a dictionary that represents a row in csv
            
            #considering different combinations when columns maybe semicolon-seperated or not
            #for each case, null checking is done
            if row[col3]!='Polypharmacy Side Effect' and row[col1]!='' and row[col2]!='':
                items1=row[col1].split('; ')
                items2=row[col2].split('; ')
                for item1 in items1:
                    for item2 in items2:
                        if item1 in dict1.keys() and item2 in dict1.keys() and item1!='' and item2!='': 
                            adj_mat[dict1[item1]][dict1[item2]]=1
                            adj_mat[dict1[item2]][dict1[item1]]=1
                        
                                              
                
    #df = pd.DataFrame(rows, columns=cols) 
  
    # Writing dataframe to csv 
    #df.to_csv('/Volumes/Farhan/Research/Code/HAN DDI/data/bio-decagon-combo_converted_Final.csv')
    return adj_mat

def create_label(path, col1, dict1, col2, dim1, dim2, FirstColHasMultiValue, SecondColHasMultiValue):
    
    labels = np.zeros((dim1,dim2))
    i=0
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
 
        
        csv_dict_reader = DictReader(read_obj)
        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:

            # considering different combinations when columns maybe semicolon-seperated or not
            # for each case, null checking is done
            if FirstColHasMultiValue==False:
              if SecondColHasMultiValue==False:
                if row[col1]!='DB09067' and row[col2]!='DB09067':
                  labels[i] = np.array([dict1[row[col1]], dict1[row[col2]], 1])
            i+=1    

    return labels

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    
    # FT 061621 download appropriate data
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)
    
    # FT 061621 load matrix
    data = sio.loadmat(data_path)
    
    # the columns referred below are liked adjacency matrices
    # they contain information on which onjects are connected
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining) conf id - 0,
    # (2) SIGMOD and VLDB papers as class 1 (database) conf id - 1 and 13,
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication) conf id - 9 and 10
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    
    # FT 061621 only considering papers which are accepted in abovementioned conferences
    p_vs_c_filter = p_vs_c[:, conf_ids]
    
    # FT 061621 extract fields, authors, terms which are associated with 
    # these conferences
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    
    
    # FT 061621 construct HIN based on these adjacency matrices/ 
    # meta-relations. these will be used to construct meta-paths
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })
    
    # FT 061621 bag of words used as feature
    features = torch.FloatTensor(p_vs_t.toarray())
    
    # FT 061621 getting dimensions of p_VS_c
    pc_p, pc_c = p_vs_c.nonzero()
    
    # FT 061621 labels 2d-matrix is generated from p_VS_c.  and it contains
    # information on which conference publishes which paper
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)
    
    # FT 061621 number of classes/nodes
    num_classes = 2
    
    # FT 061621 creating float mask which will be used for train, test and val id
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask
            
            
def load_drug_data(remove_self_loop):
    assert not remove_self_loop
    
    #drugs_dict=create_dictionary('/Volumes/Farhan/Research/Code/HAN DDI/data/drug_ids 3.txt')
    #protein_dict=create_dictionary('/Volumes/Farhan/Research/Code/HAN DDI/data/Proteins.txt')
    #CUI_dict=create_dictionary('/Volumes/Farhan/Research/Code/HAN DDI/data/CUI_ids.txt')

    PPI = pd.read_csv('/content/drive/MyDrive/HAN PPS/data/PPI_BIOGRID_FINAL_2.csv',sep=',')
    DTI = pd.read_csv('/content/drive/MyDrive/HAN PPS/data/drug_data_2nd_project_12.csv',sep=',')
    DDI = pd.read_csv('/content/drive/MyDrive/HAN PPS/data/DDI_DrugBank_4_FINAL.csv',sep=',')
    DSE = pd.read_csv('/content/drive/MyDrive/HAN PPS/data/drug_SE_FINAL.csv',sep=',')

    # Original number of interactions
    orig_ppi = len(PPI.index)
    orig_dti = len(DTI.index)
    orig_ddi = len(DDI.index)
    orig_dse = len(DSE.index)
    # ============================================================================================= #
    # REMOVING OUTLIERS
    # PPI genes
    PPI_genes = pd.unique(np.hstack((PPI['Protein 1'].values,PPI['Protein 2'].values)))
    orig_genes_ppi = len(PPI_genes) # Original number of genes


    # REDUCE DDI AND DSE DATABASES TO COMMON DRUGS ONLY
    # DDI drugs
    DDI_drugs = pd.unique(DDI[["Drug 1", "Drug 2"]].values.ravel())
    orig_drugs_ddi = len(DDI_drugs) # Original number of drugs
    orig_se_combo = len(pd.unique(DDI['Side effect Id'].values))


    # Drugs with single side effects
    DSE_drugs = pd.unique(DSE['Drug id'].values)
    orig_drug_dse = len(DSE_drugs) # Original number of drugs
    orig_se_mono = len(pd.unique(DSE['Side_effect_id']))


    # Calculate the instersection of the DDI and DSE
    # (i.e., the drugs in the interaction network that have single side effect)
    inter_drugs = np.intersect1d(DDI_drugs,DSE_drugs,assume_unique=True)
    # Choose only the entries in DDI that are in the intersection
    DDI = DDI[np.logical_and(DDI['Drug 1'].isin(inter_drugs).values,
                        DDI['Drug 2'].isin(inter_drugs).values)]


    # Some drugs in DDI that are common to all 3 datasets may only interact with genes that are
    # non-common (outsiders). That is why we need to filter a second time using this array.
    DDI_drugs = pd.unique(DDI[["Drug 1", "Drug 2"]].values.ravel())
    DSE = DSE[DSE['Drug id'].isin(DDI_drugs)]
    new_drugs_ddi = len(pd.unique(DDI[['Drug 1','Drug 2']].values.ravel()))
    new_drugs_dse = len(pd.unique(DSE['Drug id'].values))
    new_se_combo = len(pd.unique(DDI['Side effect Id'].values))
    new_se_mono = len(pd.unique(DSE['Side_effect_id']))

    # SELECT ONLY ENTRIES FROM DTI DATABASE THAT ARE PRESENT IN PREVIOUSLY REDUCED DATABASES
    orig_genes_dti = len(pd.unique(DTI['Protein'].values))
    orig_drugs_dti = len(pd.unique(DTI['ID'].values))
    DTI = DTI[np.logical_and(DTI['ID'].isin(DDI_drugs),DTI['Protein'].isin(PPI_genes))]
    DTI_genes = pd.unique(DTI['Protein'].values)
    new_genes_dti = len(DTI_genes)
    new_drugs_dti = len(pd.unique(DTI['ID'].values))
    PPI = PPI[np.logical_or(PPI['Protein 1'].isin(DTI_genes),PPI['Protein 2'].isin(DTI_genes))]
    # ============================================================================================= #
    # REDUCED DATA STRUCTURES
    # Choosing side effects. Sort DDI to be consistent with the authors
    DDI['freq'] = DDI.groupby('Side effect Id')['Side effect Id']\
                .transform('count')
    DDI = DDI.sort_values(by=['freq'], ascending=False).drop(columns=['freq'])
    se = pd.unique(DDI['Side effect Id'].values)
    se = se[:677]
    # DDI
    DDI = DDI[DDI['Side effect Id'].isin(se)].reset_index(drop=True)
    DDI_drugs = pd.unique(DDI[['Drug 1','Drug 2']].values.ravel()) # Unique drugs 
    drug2idx = {drug: i for i, drug in enumerate(DDI_drugs)}
    se_names = pd.unique(DDI['Side effect Id']) # Unique joint side effects
    se_combo_name2idx = {se: i for i, se in enumerate(se_names)}
    n_drugs = len(DDI_drugs)
    # DSE
    DSE = DSE[DSE['Drug id'].isin(DDI_drugs)].reset_index(drop=True)
    dse_drugs = len(pd.unique(DSE['Drug id'].values))
    se_mono_names = pd.unique(DSE['Side_effect_id'].values) # Unique individual side effects
    se_mono_name2idx = {name: i for i, name in enumerate(se_mono_names)}
    n_semono = len(se_mono_names)
    # DTI
    DTI = DTI[DTI['ID'].isin(DDI_drugs)].reset_index(drop=True)
    DTI_genes = pd.unique(DTI['Protein']) # Unique genes in DTI
    DTI_drugs = pd.unique(DTI['ID']) # Unique drugs in DTI
    dti_drugs = len(DTI_drugs)
    dti_genes = len(DTI_genes)
    # PPI
    PPI = PPI[np.logical_or(PPI['Protein 1'].isin(DTI_genes),
                          PPI['Protein 2'].isin(DTI_genes))].reset_index(drop=True)
    PPI_genes = pd.unique(PPI[['Protein 1','Protein 2']].values.ravel()) # Unique genes in PPI
    gene2idx = {gene: i for i, gene in enumerate(PPI_genes)}
    n_genes = len(PPI_genes)
    # ============================================================================================= #
    # ADJACENCY MATRICES AND DEGREES
    # DDI
    ddi_adj = np.zeros([n_drugs,n_drugs],dtype=int)
    for i in DDI.index:
        row = drug2idx[DDI.loc[i,'Drug 1']]
        col = drug2idx[DDI.loc[i,'Drug 2']]
        ddi_adj[row,col] = 1
        ddi_adj[col,row] = 1
    ddi_adj = np.array(ddi_adj)
    
    # DTI
    dti_adj = np.zeros([n_drugs,n_genes],dtype=int)
    for i in DTI.index:
        row = drug2idx[DTI.loc[i,'ID']]
        col = gene2idx[DTI.loc[i,'Protein']]
        dti_adj[row,col] = 1
    dti_adj = np.array(dti_adj)
    
    # PPI
    ppi_adj = np.zeros([n_genes,n_genes],dtype=int)
    for i in PPI.index:
        row = gene2idx[PPI.loc[i,'Protein 1']]
        col = gene2idx[PPI.loc[i,'Protein 2']]
        ppi_adj[row,col]=ppi_adj[col,row]=1
    ppi_adj = np.array(ppi_adj)

    # DSE
    dse_adj = np.zeros([n_drugs,n_semono],dtype=int)
    for i in DSE.index:
        row = drug2idx[DSE.loc[i,'Drug id']]
        col = se_mono_name2idx[DSE.loc[i,'Side_effect_id']]
        dse_adj[row,col] = 1
    dse_adj = np.array(dse_adj)

    # Drug Feature matrix
    d_chem_adj = np.zeros([n_drugs,167],dtype=int)
    for i in DTI.index:
        row = drug2idx[DTI.loc[i,'ID']]
        for j in range(167):
            d_chem_adj[row][j] = DTI.loc[i,'Binary Vector'][j+1]
    d_chem_adj = np.array(d_chem_adj)


    # Drug Feature matrix
    drug_feat = np.zeros([n_drugs,32],dtype=int)
    for i in DTI.index:
        row = drug2idx[DTI.loc[i,'ID']]
        for j in range(32):
            drug_feat[row][j] = DTI.loc[i,'EPFS SMILES'][j+1]
    drug_feat=np.array(drug_feat)
    drug_feat = normalize_features(drug_feat)
    #drug_feat = torch.FloatTensor(drug_feat).to(device)

    # Protein Feature matrix
    gene_feat = np.zeros([n_genes,58],dtype=int)
    for i in DTI.index:
        row = gene2idx[DTI.loc[i,'Protein']]
        for j in range(38):
            gene_feat[row][j] = DTI.loc[i,'EPFS ACS'][j+1]
    gene_feat = np.array(gene_feat)
    gene_feat = normalize_features(gene_feat)
    #gene_feat = torch.FloatTensor(gene_feat).to(device)


    # Side effect Feature matrix
    se_feat = np.zeros([n_semono,n_semono],dtype=int)
    for key, val in se_mono_name2idx.items():
        for j in range(n_semono):
            if val==j:
                se_feat[val][j]=1
            else:
                se_feat[val][j]=0
    se_feat = np.array(se_feat)
    se_feat = normalize_features(se_feat)
    #se_feat = torch.FloatTensor(se_feat).to(device)

    # Chemical structure Feature matrix
    chem_struct_feat = np.zeros([167,167],dtype=int)
    for i in range(167):
        for j in range(167):
            if i==j:
                chem_struct_feat[i][j]=1
            else:
                chem_struct_feat[i][j]=0
    chem_struct_feat = np.array(chem_struct_feat)
    chem_struct_feat = normalize_features(chem_struct_feat)
    #chem_struct_feat = torch.FloatTensor(chem_struct_feat).to(device)

    # FT 061621 construct HIN based on these adjacency matrices/ 
    # meta-relations. these will be used to construct meta-paths
    g = dgl.heterograph({
        ('drug', 'ddi', 'drug'): ddi_adj.nonzero(),
        ('drug', 'dp', 'protein'): dti_adj.nonzero(),
        ('protein', 'pd', 'drug'): dti_adj.transpose().nonzero(),
        ('protein', 'pp', 'protein'): ppi_adj.nonzero(),
        ('drug', 'ds', 'side_effect'): dse_adj.nonzero(),
        ('side_effect', 'sd', 'drug'): dse_adj.transpose().nonzero(),
        ('drug', 'dcs', 'chem_Struct'): d_chem_adj.nonzero(),
        ('chem_Struct', 'csd', 'drug'): d_chem_adj.transpose().nonzero()
    })


    g.nodes['drug'].data['hv'] =  torch.from_numpy(drug_feat)
    g.nodes['protein'].data['hv'] = torch.from_numpy(gene_feat)
    g.nodes['chem_Struct'].data['hv'] = torch.from_numpy(chem_struct_feat)
    g.nodes['side_effect'].data['hv'] = torch.from_numpy(se_feat)
    
    #g.edges['ddi'].data['he'] = ddi_feat
    return g
            
    

def load_data(remove_self_loop=False):
    return load_drug_data(remove_self_loop)
    #if dataset == 'ACM':
    #    return load_acm(remove_self_loop)
    #elif dataset == 'ACMRaw':
    #    return load_acm_raw(remove_self_loop)
    #else:
    #    return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))