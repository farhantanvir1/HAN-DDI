#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:50:45 2021

@author: illusionist
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Imports original DECAGON database and translates it into adjaceny matrices and enumeration 
dictionaries. First the original dataset is filtered so it has no unlinked nodes creating a 
consistent network. Then a fraction of the dataset is chosen, selecting a fixed number of 
polypharmacy side effects given by parameter N (defaults to 964). With the reduced network, 
the adjacency matrices and the node enumeration dictionaries are created and exported as a 
pickle python3 readable file.

Parameters
----------
number of side effects : int, default=964
    Number of joint drug side effects to be chosen from the complete dataset. If not given, 
    the program uses the maximum number of side effects used by the authors of DECAGON.
"""
# ============================================================================================= #
import argparse
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from joblib import Parallel, delayed
parser = argparse.ArgumentParser(description='Remove outliers from datasets')
parser.add_argument('N', nargs='?',default =964,type=int, help="Number of side effects")
args = parser.parse_args()
N = args.N
# Import databases as pandas dataframes
PPI = pd.read_csv('/Volumes/Farhan/Research/Code/HAN DDI/data/new data/FInal/PPI_BIOGRID_FINAL_2.csv',sep=',')
DTI = pd.read_csv('/Volumes/Farhan/Research/Code/HAN DDI/data/new data/FInal/drug_data_2nd_project_12.csv',sep=',')
DDI = pd.read_csv('/Volumes/Farhan/Research/Code/HAN DDI/data/new data/FInal/DDI_DrugBank_4_FINAL.csv',sep=',')
DSE = pd.read_csv('/Volumes/Farhan/Research/Code/HAN DDI/data/new data/FInal/drug_SE_FINAL.csv',sep=',')
print('\nData loaded\n')
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
print('Outliers removed\n')
# ============================================================================================= #
# REDUCED DATA STRUCTURES
# Choosing side effects. Sort DDI to be consistent with the authors
DDI['freq'] = DDI.groupby('Side effect Id')['Side effect Id']\
            .transform('count')
DDI = DDI.sort_values(by=['freq'], ascending=False).drop(columns=['freq'])
se = pd.unique(DDI['Side effect Id'].values)
se = se[:N]
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
print('Side effects selected\n')
# ============================================================================================= #
# ADJACENCY MATRICES AND DEGREES
# DDI
def se_adj_matrix(se_name):
    m = np.zeros([n_drugs,n_drugs],dtype=int)
    seDDI = DDI[DDI['Side effect Id'].str.match(se_name)].reset_index()
    for j in seDDI.index:
        row = drug2idx[seDDI.loc[j,'Drug 1']]
        col = drug2idx[seDDI.loc[j,'Drug 2']]
        m[row,col] = m[col,row] = 1
    return sp.csr_matrix(m) 
ddi_adj_list = Parallel(n_jobs=8)\
    (delayed(se_adj_matrix)(d) for d in se_combo_name2idx.keys())        
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
print('DDI adjacency matrices generated\n')
# DTI
dti_adj = np.zeros([n_genes,n_drugs],dtype=int)
for i in DTI.index:
    row = gene2idx[DTI.loc[i,'Protein']]
    col = drug2idx[DTI.loc[i,'ID']]
    dti_adj[row,col] = 1
dti_adj = sp.csr_matrix(dti_adj)
print('DTI adjacency matrix generated\n')
# PPI
ppi_adj = np.zeros([n_genes,n_genes],dtype=int)
for i in PPI.index:
    row = gene2idx[PPI.loc[i,'Protein 1']]
    col = gene2idx[PPI.loc[i,'Protein 2']]
    ppi_adj[row,col]=ppi_adj[col,row]=1
ppi_degrees = np.sum(ppi_adj,axis=0)
ppi_adj = sp.csr_matrix(ppi_adj)
print('PPI adjacency matrix generated\n')


# DSE
dse_adj = np.zeros([n_drugs,n_semono],dtype=int)
for i in DSE.index:
    row = drug2idx[DSE.loc[i,'Drug id']]
    col = se_mono_name2idx[DSE.loc[i,'Side_effect_id']]
    dse_adj[row,col] = 1
dse_adj = sp.csr_matrix(dse_adj)
print('DSE adjacency matrix generated\n')

# Drug Feature matrix
drug_feat = np.zeros([n_drugs,38],dtype=int)
for i in DTI.index:
    row = drug2idx[DTI.loc[i,'ID']]
    for j in range(38):
        drug_feat[row][j] = DTI.loc[i,'EPFS SMILES'][j+1]
drug_feat = sp.csr_matrix(drug_feat)
print('Drug feature generated\n')

# Protein Feature matrix
gene_feat = np.zeros([n_genes,38],dtype=int)
for i in DTI.index:
    row = gene2idx[DTI.loc[i,'Protein']]
    for j in range(38):
        gene_feat[row][j] = DTI.loc[i,'EPFS ACS'][j+1]
gene_feat = sp.csr_matrix(gene_feat)
print('Protein feature generated\n')

# ============================================================================================= #
# CONTROL PRINTING
# Interactions (edges)
print('==== CHANGES MADE IN DATA ====')
print('Interactions (edges)')
print ('Original number of PPI interactions',orig_ppi)
print ('New number of PPI interactions',len(PPI.index))
print('\n')
print ('Original number of DTI interactions',orig_dti)
print ('New number of DTI interactions',len(DTI.index))
print('\n')
print ('Original number of DDI interactions',orig_ddi)
print ('New number of DDI interactions', len(DDI.index))
print('\n')
print ('Original number of DSE interactions',orig_dse)
print('New number of DSE interactions',len(DSE.index))
print('\n')
# Drugs and genes (nodes)
print('Drugs and genes (nodes)')
print("Original number of drugs in DSE:",orig_drug_dse)
print("New number of drugs in DSE:", dse_drugs)
print('\n')
print("Original number drugs in DTI",orig_drugs_dti)
print("New number of drugs in DTI",dti_drugs)
print('\n')
print('Original number of genes in DTI:',orig_genes_dti)
print('New number of genes in DTI:',dti_genes)
print('\n')
print('Original number of genes in PPI:',orig_genes_ppi)
print('New number of genes in PPI:',n_genes)
print('\n')
print('Original number of drugs in DDI:',orig_drugs_ddi)
print('New number of drugs in DDI:',n_drugs)
print('\n')
# Side effects
print('Side effects')
print('Original number of joint side effects:',orig_se_combo)
print('New number of joint side effects:', len(se_names))
print('\n')
print('Original number of single side effects:', orig_se_mono)
print('New number of single side effects:', n_semono)
# ============================================================================================= #
# SAVING DATA STRUCTURES
data = {}
# Dictionaries
data['gene2idx'] = gene2idx
data['drug2idx'] = drug2idx
data['se_mono_name2idx'] = se_mono_name2idx
data['se_combo_name2idx'] = se_combo_name2idx
# DDI
data['ddi_adj_list'] = ddi_adj_list
data['ddi_degrees_list'] = ddi_degrees_list
# DTI
data['dti_adj'] = dti_adj
# PPI
data['ppi_adj'] = ppi_adj
data['ppi_degrees'] = ppi_degrees

data['dse_adj'] = dse_adj
# DSE
data['drug_feat'] = drug_feat
data['gene_feat'] = gene_feat

hg = dgl.heterograph({
        ('drug', 'dp', 'protein'): dti_adj.nonzero(),
        ('protein', 'pp', 'protein'): ppi_adj.nonzero(),
        ('protein', 'ppt', 'protein'): ppi_adj.nonzero(),
        ('protein', 'pd', 'drug'): dti_adj.transpose().nonzero(),
        ('drug', 'ds', 'side_effect'): dse_adj.nonzero(),
        ('side_effect', 'sd', 'drug'): dse_adj.transpose().nonzero()
    })
    
print('meta-path and graph created')

# Exporting
filename = './' + str(n_semono) +\
           '_genes_' + str(n_genes) + '_drugs_' + str(n_drugs) + '_se_' + str(N)
print('Output_file: ',filename,'\n')
with open(filename, 'wb') as f:
    pickle.dump(data, f, protocol=3)


