import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GINConv

L_Relu = nn.LeakyReLU()                                                                                                                                     
sig = nn.Sigmoid()
Relu = nn.ReLU()
tanh = nn.Tanh()

class SPIN(nn.Module): #aggregation choices = ["concat","sum"]
  def __init__(self,d,d_,num_classes,r,dim1,dropout_p=0.5,agg = 'concat',attention=True):
    super(SPIN, self).__init__()
    self.d = d
    self.d_ = d_
    self.num_classes = num_classes
    self.agg = agg
    self.conv_layers = nn.ModuleList()
    self.att_layers = nn.ModuleList()
    self.branches = r+1
    self.attention = attention
    for i in range(self.branches):
      #l = nn.Sequential(nn.Linear(self.d,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,self.d_),nn.LeakyReLU())
      l = nn.Sequential(nn.Linear(self.d,dim1),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.LeakyReLU(),nn.Linear(dim1,self.d_),nn.LeakyReLU())
      self.conv_layers.append(l)
    if (attention):
      for i in range(self.branches):
        a = nn.Linear(self.d_,1,bias = False)
        self.att_layers.append(a)
    
    cl_dim1 = 32
    if (agg == 'concat'):
      self.classify = nn.Sequential(nn.Linear(self.branches*d_,cl_dim1),nn.LeakyReLU(),nn.Linear(cl_dim1,cl_dim1),nn.LeakyReLU(),nn.Linear(cl_dim1,self.num_classes))
    else:
      self.classify = nn.Sequential(nn.Linear(d_,cl_dim1),nn.LeakyReLU(),nn.Linear(cl_dim1,cl_dim1),nn.LeakyReLU(),nn.Linear(cl_dim1,self.num_classes))
  
  def attend_nodes(self,alpha_mat,Y):
    S = torch.empty((self.d_,self.branches))
    for i in range(self.branches):
      S[:,i] = (alpha_mat[:,i].reshape(-1,1)*Y[i]).sum(dim=0)
    return S
    #return alpha.reshape(-1,1)*Y
  
  def get_attention_scores(self,Y):
    w = torch.empty((len(Y[0]),self.branches))
    for i in range(self.branches):
      w[:,i] = F.softmax(Relu(self.att_layers[i](Y[i])),dim=0).squeeze()
    self.weights = w
    return self.weights

  def get_S(self,Y):
    S = torch.empty((self.d_,self.branches))
    for i in range(self.branches):
      S[:,i] = Y[i].sum(dim=0).squeeze()
    return S

  def forward(self,l): #l = [X_list,AX_list,A2X_list,...,label_batches,graph_batches]
    l = l[:-3] #dont need the labels and the actual graphs and last added summation nodes
    C = torch.empty((len(l[0]),self.num_classes))
    for i in range(len(l[0])):
      Y = [] #Y = [Y0,Y1,...]
      for j in range(self.branches):
        Y.append(self.conv_layers[j](l[j][i]))
      
      if (self.attention):
        alpha_mat = self.get_attention_scores(Y)
        S = self.attend_nodes(alpha_mat,Y)
      else:
        S = self.get_S(Y)

      if (self.agg == 'concat'): em = S.reshape(1,-1)
      else: em = S.sum(dim=1).reshape(1,-1)
    
      cl = self.classify(em)
      C[i] = cl
    return C

class GraphSage(nn.Module):
  def __init__(self,d,dim1,r,num_classes):
    super(GraphSage, self).__init__()
    self.d = d
    self.r = r
    self.dim1 = dim1
    self.num_classes = num_classes
    self.layers = [SAGEConv(d,self.dim1,'mean')]
    self.layers += [SAGEConv(self.dim1,self.dim1,'mean') for _ in range(r-1)]
    self.layers = nn.ModuleList(self.layers)

    self.classify = nn.Sequential(nn.Linear(self.dim1,self.dim1),nn.LeakyReLU(),nn.Linear(self.dim1,num_classes))
    
  def forward(self,l): #l = [X_list,AX_list,A2X_list,...,label_batches,graph_batches,_]
    X_list = l[0]
    graph_list = l[-2]
    C = torch.empty((len(X_list),self.num_classes))
    for i in range(len(X_list)):
      Y = X_list[i]
      for j in range(self.r):
        Y = L_Relu(self.layers[j](graph_list[i],Y))

      em = Y.sum(dim=0).reshape(1,-1)
      cl = self.classify(em)
      
      C[i] = cl
    return C

class GIN(nn.Module):
  def __init__(self,d,dim1,r,num_classes):
    super(GIN, self).__init__()
    self.d = d
    self.r = r
    self.dim1 = dim1
    self.num_classes = num_classes
    self.MLP_layers = [nn.Sequential(nn.Linear(self.d,self.dim1),nn.LeakyReLU(),nn.Linear(self.dim1,self.dim1),nn.LeakyReLU())]
    self.MLP_layers += [nn.Sequential(nn.Linear(self.dim1,self.dim1),nn.LeakyReLU(),nn.Linear(self.dim1,self.dim1),nn.LeakyReLU()) for _ in range(r-1)]
    self.MLP_layers = nn.ModuleList(self.MLP_layers)

    self.GIN_layers = [GINConv(self.MLP_layers[i],'sum',init_eps=0.1) for i in range(r)]
    self.GIN_layers = nn.ModuleList(self.GIN_layers)

    self.classify = nn.Sequential(nn.Linear(self.dim1,self.dim1),nn.LeakyReLU(),nn.Linear(self.dim1,num_classes))
    
  def forward(self,l): #l = [X_list,AX_list,A2X_list,...,label_batches,graph_batches,_]
    X_list = l[0]
    graph_list = l[-2]
    C = torch.empty((len(X_list),self.num_classes))
    for i in range(len(X_list)):
      Y = X_list[i]
      for j in range(self.r):
        Y = L_Relu(self.GIN_layers[j](graph_list[i],Y))

      em = Y.sum(dim=0).reshape(1,-1)
      cl = self.classify(em)
      
      C[i] = cl
    return C

class MLP_classifier(nn.Module):
  def __init__(self,d,num_layers,dim1,num_classes):
    super(MLP_classifier, self).__init__()
    self.d = d
    self.dim1 = dim1
    self.num_classes = num_classes
    self.MLP_layers = [nn.Sequential(nn.Linear(self.d,self.dim1),nn.LeakyReLU())]
    self.MLP_layers += [nn.Sequential(nn.Linear(self.dim1,self.dim1),nn.LeakyReLU()) for _ in range(num_layers)] 
    self.MLP_layers = nn.ModuleList(self.MLP_layers)

    self.classify = nn.Sequential(nn.Linear(self.dim1,self.dim1),nn.LeakyReLU(),nn.Linear(self.dim1,num_classes))
    
  def forward(self,l): #l = [X_list,AX_list,A2X_list,...,label_batches,graph_batches,_]
    X = l[-1]
    for layer in self.MLP_layers:
      X = layer(X)

    cl = self.classify(X)
      
    return cl
