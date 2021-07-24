pip install dgl

import time
import dgl
import networkx as nx
from scipy.sparse import random, coo_matrix 
import numpy as np
from models import *

class synthetic_dataset():
  def __init__(self,num_nodes,num_graphs,d,edge_prob=0.1):
    self.num_nodes = num_nodes
    self.num_graphs = num_graphs
    self.p = edge_prob
    self.graphs = self.create_graphs()
    self.input_feat_dim = d
    #print ("Graphs created | details:")
    #print ("Num of graphs:",len(self.graphs))
    #print ("Num of nodes in each graph:",self.num_nodes)
    #print ("Input feat dim (d):",self.input_feat_dim)
  
  def create_graphs(self):
    return [dgl.from_networkx(nx.gnp_random_graph(self.num_nodes, self.p, seed=i, directed=False)) for i in range(self.num_graphs)]

def count_edges(data):
  edges = []
  for g in data.graphs:
    A = g.adjacency_matrix(False, scipy_fmt="csr").toarray()
    edges.append(sum(sum(A))//2)
  print (sum(edges)/len(edges))
  print (np.std(edges))

def get_embed_X(graph):
  X = random(graph.num_nodes(), data.input_feat_dim, density=0.1, format='coo')
  return coo_matrix(X)

def create_batches(graphs_list,precomputations_needed = False):  #precomputationsa are not needed incase of sequential models
  X_list,AX_list,A2X_list,A3X_list = [],[],[],[]
  for graph in graphs_list:
    X_list.append(get_embed_X(graph))
  if (precomputations_needed):
    AX_list,A2X_list,A3X_list = get_precomputions(graphs_list,X_list)
  return X_list,AX_list,A2X_list,A3X_list

def get_precomputions(graphs,X): #get AX and A2X lists
  AX_list,A2X_list,A3X_list = [],[],[]
  for i in range(len(X)):
    A = graphs[i].adjacency_matrix(False, scipy_fmt="csr").toarray()
    ax = coo_matrix(A*X[i])
    a2x = coo_matrix(A*ax)
    a3x = coo_matrix(A*a2x)
    AX_list.append(ax)
    A2X_list.append(a2x)
    A3X_list.append(a3x)
  return AX_list,A2X_list,A3X_list

def sparse_tensorize(lst):
  return [get_sparse_tensor_from_scipy_coo(t) for t in lst]

def dense_tensor_from_sparse(t):
  return t.to_dense()

def get_sparse_tensor_from_scipy_coo(coo):
  values = coo.data
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = coo.shape
  return torch.sparse.FloatTensor(i, v, torch.Size(shape))

num_classes = 2
num_nodes = 100
num_graphs = 100
input_dim = 50
p = 1
data = synthetic_dataset(num_nodes,num_graphs,input_dim,edge_prob=p)

X_list,AX_list,A2X_list,A3X_list = create_batches(data.graphs,precomputations_needed = 1)
X_list,AX_list,A2X_list,A3X_list = map(sparse_tensorize, [X_list,AX_list,A2X_list,A3X_list])
X_list_dense = list(map(dense_tensor_from_sparse,X_list))
labels = torch.randint(2,size=(num_graphs,1)).squeeze()
count_edges(data)

d = data.input_feat_dim #fixed
r_hop = 3 #hyperpara
#net = SPIN(d,d,num_classes,r_hop,d,dropout_p=0.5,agg = 'concat',attention=True)
net = GIN(d,r_hop,num_classes)
#net = GraphSage(d,r_hop,num_classes)

num_fwd_passes = 2
time_counter = 0 
for _ in range(num_fwd_passes):
  CE_loss = nn.CrossEntropyLoss()
  start = time.process_time()
  #l = net([X_list,AX_list,A2X_list,A3X_list])
  l = net(X_list_dense,data.graphs)
  loss = CE_loss(l,labels.type(torch.LongTensor)) 
  loss.backward()
  time_counter += (time.process_time() - start)
print ("n = ",num_nodes)
print ("Average time taken for the forward pass of %d graphs is %f"%(len(data.graphs),time_counter/num_fwd_passes),end=' ')