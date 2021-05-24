import dgl #pip install dgl if needed
import json
from sklearn.model_selection import train_test_split
from collections import OrderedDict, defaultdict, Counter
import random
import numpy as np
from scipy.sparse import coo_matrix
import torch

def get_sparse_tensor_from_scipy_coo(coo):
  values = coo.data
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = coo.shape
  return torch.sparse.FloatTensor(i, v, torch.Size(shape))

Enzymes = {
    "name":"ENZYMES",
    "category":"bioinformatics",
    "input_feat_dim":21,
    "num_classes":6
}
DD = {
    "name":"DD",
    "category":"bioinformatics",
    "input_feat_dim":89,
    "num_classes":2
}
NCI1 = {
    "name":"NCI1",
    "category":"bioinformatics",
    "input_feat_dim":37,
    "num_classes":2
}
Proteins = {
    "name":"PROTEINS_full",
    "category":"bioinformatics",
    "input_feat_dim":3,
    "num_classes":2
}
IMDB_B = {
    "name":"IMDB-BINARY",
    "category":"social",
    "input_feat_dim":1,
    "num_classes":2
}
IMDB_M = {
    "name":"IMDB-MULTI",
    "category":"social",
    "input_feat_dim":1,
    "num_classes":3
}
Collab = {
    "name":"COLLAB",
    "category":"social",
    "input_feat_dim":1,
    "num_classes":3
}
Reddit_B = {
    "name":"REDDIT-BINARY",
    "category":"social",
    "input_feat_dim":1,
    "num_classes":2
}
Reddit_M = {
    "name":"REDDIT-MULTI-5K",
    "category":"social",
    "input_feat_dim":1,
    "num_classes":5
}

dataset_dict = {
    "ENZYMES":Enzymes,
    "DD":DD,
    "NCI1":NCI1,
    "PROTEINS_full":Proteins,
    "IMDB-BINARY":IMDB_B,
    "IMDB-MULTI":IMDB_M,
    "COLLAB":Collab,
    "REDDIT-BINARY":Reddit_B,
    "REDDIT-MULTI-5K":Reddit_M
}

class dataset():
  def __init__(self,name,category,num_classes,load_splits = False,predefined_splits_filepath = ''):
    #json_filename: provide the predefined data splits json filepath
    self.name = name
    self.category = category
    self.num_classes = num_classes
    data = dgl.data.TUDataset(self.name)
    self.labels = [lab.item() for lab in data.graph_labels]
    self.graphs =  data.graph_lists
    if (category == "social"):
      #self.input_feat_dim = self.highest_degree()
      self.input_feat_dim = 1
    else:
      self.input_feat_dim = 21 if (name=="ENZYMES") else self.num_atoms()+1
    
    self.load_splits = load_splits
    if (load_splits):
      self.graph_sets, self.label_sets = self.load_splits_from_json(predefined_splits_filepath)
    else:
      self.graph_sets, self.label_sets = create_10_fold_splits(self.graphs,self.labels,random_seed=17)
    self.stratification_ratios = list(self.get_stratification_ratios(self.labels).values())

  def get_stratification_ratios(self,lst):
    x = OrderedDict(Counter(lst))
    ln = len(lst)
    for key in x.keys():
      x[key] = x[key]/ln
    return x
  
  def load_splits_from_json(self,json_filename): 
    #loads all the fold splits from the predefined JSON file
    with open(json_filename, 'r') as j:
      self.content = json.loads(j.read()) #list of dict corresponding to each fold
    graph_sets = []
    lab_sets = []
    for fold in self.content:
      indices = fold['test']
      graph_sets.append([self.graphs[indx] for indx in indices])
      lab_sets.append([self.labels[indx] for indx in indices])
    return graph_sets,lab_sets

  def num_atoms(self):
    return max(max(g.ndata['node_labels']) for g in self.graphs)

def split_data(datasetObj,fold_index_k=0,val_split = 0.1,validation_split_seed=17):
  #fold_index_k = 2 #index between [0,10)
  #gets us the desired fold as testing set
  dataFold = dataset_split_class(datasetObj,fold_index_k=fold_index_k,val_split = 0.1,validation_split_seed=17)
  return dataFold

class dataset_split_class():
  def __init__(self,datasetObj,fold_index_k=0,val_split = 0.1,validation_split_seed=17):
    if (datasetObj.load_splits):
      train_indices = datasetObj.content[fold_index_k]['model_selection'][0]['train']
      val_indices = datasetObj.content[fold_index_k]['model_selection'][0]['validation']
      test_indices = datasetObj.content[fold_index_k]['test']
      self.graphs_train = [datasetObj.graphs[indx] for indx in train_indices]
      self.lab_train = [datasetObj.labels[indx] for indx in train_indices]
      self.graphs_val = [datasetObj.graphs[indx] for indx in val_indices]
      self.lab_val = [datasetObj.labels[indx] for indx in val_indices]
      self.graphs_test = [datasetObj.graphs[indx] for indx in test_indices]
      self.lab_test = [datasetObj.labels[indx] for indx in test_indices]
    else:
      self.graphs_test = datasetObj.graph_sets[fold_index_k]
      self.lab_test = datasetObj.label_sets[fold_index_k]
      self.graphs_train,self.lab_train = [],[]
      for st in datasetObj.graph_sets[:fold_index_k]+datasetObj.graph_sets[fold_index_k+1:]:
        for graph in st:
          self.graphs_train.append(graph)
      for st in datasetObj.label_sets[:fold_index_k]+datasetObj.label_sets[fold_index_k+1:]:
        for label in st:
          self.lab_train.append(label)
      self.graphs_train, self.graphs_val, self.lab_train, self.lab_val = train_test_split(self.graphs_train, self.lab_train, test_size=val_split, random_state=validation_split_seed)
    
  def create_batches(self,datasetObj,r_hop=2,batch_size=32,save_batches = False,path = ''):
    #batches = [X_batches,AX_batches,...,ArX_batches,label_batches,graph_batches]
    self.train_batches = batch(self.graphs_train,self.lab_train,batch_size,datasetObj.category,datasetObj.num_classes,r_hop,datasetObj.input_feat_dim,stratification_ratios = datasetObj.stratification_ratios,save_batches=False,path = '')
    self.val_batches = batch(self.graphs_val,self.lab_val,len(self.lab_val),datasetObj.category,datasetObj.num_classes,r_hop,datasetObj.input_feat_dim,save_batches=save_batches,path = '')
    self.test_batches = batch(self.graphs_test,self.lab_test,len(self.lab_test),datasetObj.category,datasetObj.num_classes,r_hop,datasetObj.input_feat_dim,save_batches=save_batches,path = '')
    self.num_batches = len(self.train_batches[0])

def batch(graphs_list,labels,batch_size,category,num_classes,r,input_feat_dim,stratification_ratios = [],save_batches = False,path = ''):
  num_batches = len(graphs_list)//batch_size
  X_batches, label_batches, graph_batches = [],[],[]
  all_batches = [[] for _ in range(r+3)]
  indices_dict = defaultdict()
  #len of stratification_ratios will give me the num_classes
  for i in range(num_classes):
    indices_dict[i] = []
  for i in range(len(labels)):
    indices_dict[labels[i]].append(i)
  sample_size = [round(ratio*batch_size) for ratio in stratification_ratios[:num_classes-1]]
  sample_size.append(batch_size-sum(sample_size))
  for i in range(num_batches):
    if (batch_size == len(labels)):
      indices = [i for i in range(batch_size)]
    else:
      indices = []
      for j in range(num_classes):
        indices+= random.sample(indices_dict[j],sample_size[j])
    graph_batches.append([graphs_list[index] for index in indices])
    X_batches.append([get_embed_X(graphs_list[index],category,input_feat_dim) for index in indices])
    c = get_precomputions([graphs_list[index] for index in indices],X_batches[i],r,category)
    f = [torch.Tensor(t) for t in X_batches[i]] if (category == "social") else [get_sparse_tensor_from_scipy_coo(t) for t in X_batches[i]]
    all_batches[0].append(f)
    for j in range(1,r+1):
      all_batches[j].append(c[j-1])
    label_batches.append(torch.tensor([labels[index] for index in indices]))
    all_batches[r+1].append(label_batches[i])
    all_batches[r+2].append(graph_batches[i])
  if (save_batches): save_variable(all_batches,path)
  return all_batches

def get_precomputions(graphs,X,r,category):
  all_list = [[] for _ in range(r)] #it should be like a list [AX_list,A2X_list,A3X_list,A4X_list]
  if (category == 'social'):
    for i in range(len(X)):
      A = get_adjacency(graphs[i]) #this is the normalized adjacency
      c = []
      x = X[i]
      for _ in range(r):
        x = A@x
        c.append(x) #c is like [ax,a2x,a3x]
      c = [torch.Tensor(t) for t in c]
      for j in range(r):
        all_list[j].append(c[j])
  else:
    for i in range(len(X)):
      A = get_adjacency(graphs[i]) #this is the normalized adjacency
      c = []
      x = X[i]
      for _ in range(r):
        x = coo_matrix(A*x)
        c.append(x) #c is like [ax,a2x,a3x]
      c = [get_sparse_tensor_from_scipy_coo(t) for t in c]
      get_sparse_tensor_from_scipy_coo
      for j in range(r):
        all_list[j].append(c[j])
  return all_list 

def get_adjacency(graph): #input = DGL hetero graph
  A = graph.adjacency_matrix(False, scipy_fmt="csr").toarray()
  D = np.zeros((graph.num_nodes(),graph.num_nodes())) #this is d_inverse_1/2
  degrees = graph.in_degrees()
  #if (0 in degrees): print ("Found zero degree node")
  for i in range(len(degrees)):
    D[i,i] = (1/degrees[i])**0.5 if (degrees[i]!=0) else 0
  A_tilde = np.matmul(np.matmul(D,A),D)
  return A_tilde
  
def get_embed_X(graph,category,input_feat_dim):
  if (category == 'social'): #for social dataset, endcoding the degree info
    degrees = graph.in_degrees()//2
    X = np.array(degrees).reshape(graph.num_nodes(),1)
  else: #for bioinformatics dataset, encoding the atom type
    X = np.zeros((graph.num_nodes(),input_feat_dim))
    atoms = graph.ndata['node_labels']
    for i in range(graph.num_nodes()):
      X[i,atoms[i]] = 1
    if (input_feat_dim == 21): #special case for ENZYMES
      X1 = np.array(graph.ndata["node_attr"]) #nx18 matrix
      X = np.concatenate((X1,X[:,:3]),axis=1) #converting it to nx21
  if (category != "social") : X = coo_matrix(X)
  return X

def create_10_fold_splits(graphs,labels,random_seed=17):
  #create the k-fold splits and save them, so that we can even use the same anytime
  #using the unconventional train-test split method for creating 10 folds as it has the "random seed" that could help us reproduce this randomness
  graphs_80, graphs_20, lab_80, lab_20 = train_test_split(graphs, labels, test_size=0.2, random_state=random_seed)

  graphs_set_1,graphs_set_2,lab_set_1,lab_set_2 = train_test_split(graphs_20, lab_20, test_size=0.5, random_state=random_seed)

  graphs_40_1, graphs_40_2, lab_40_1, lab_40_2 = train_test_split(graphs_80, lab_80, test_size=0.5, random_state=random_seed)

  graphs_20_1, graphs_20_2, lab_20_1, lab_20_2 = train_test_split(graphs_40_1, lab_40_1, test_size=0.5, random_state=random_seed)
  graphs_set_3,graphs_set_4,lab_set_3,lab_set_4 = train_test_split(graphs_20_1, lab_20_1, test_size=0.5, random_state=random_seed)
  graphs_set_5,graphs_set_6,lab_set_5,lab_set_6 = train_test_split(graphs_20_2, lab_20_2, test_size=0.5, random_state=random_seed)

  graphs_20_1, graphs_20_2, lab_20_1, lab_20_2 = train_test_split(graphs_40_2, lab_40_2, test_size=0.5, random_state=random_seed)
  graphs_set_7,graphs_set_8,lab_set_7,lab_set_8 = train_test_split(graphs_20_1, lab_20_1, test_size=0.5, random_state=random_seed)
  graphs_set_9,graphs_set_10,lab_set_9,lab_set_10 = train_test_split(graphs_20_2, lab_20_2, test_size=0.5, random_state=random_seed)

  graph_sets = [graphs_set_1,graphs_set_2,graphs_set_3,graphs_set_4,graphs_set_5,graphs_set_6,graphs_set_7,graphs_set_8,graphs_set_9,graphs_set_10]
  label_sets = [lab_set_1,lab_set_2,lab_set_3,lab_set_4,lab_set_5,lab_set_6,lab_set_7,lab_set_8,lab_set_9,lab_set_10]

  return graph_sets,label_sets

def load_dataset(dataset_name,load_datasplits = False,predefined_splits_filepath = ''):
  data_dct = dataset_dict[dataset_name]
  data_obj = dataset(dataset_name,data_dct['category'],data_dct['num_classes'],load_splits = load_datasplits,predefined_splits_filepath = predefined_splits_filepath)
  return data_obj