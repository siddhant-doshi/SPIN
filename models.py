
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
    self.conv_layers = []
    self.att_layers = []
    self.branches = r+1
    self.attention = attention
    for i in range(self.branches):
      #l = nn.Sequential(nn.Linear(self.d,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.Dropout(p=dropout_p),nn.LeakyReLU(),nn.Linear(dim1,self.d_),nn.LeakyReLU())
      l = nn.Sequential(nn.Linear(self.d,dim1),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.LeakyReLU(),nn.Linear(dim1,dim1),nn.LeakyReLU(),nn.Linear(dim1,self.d_),nn.LeakyReLU())
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
      w[:,i] = F.softmax(self.att_layers[i](Y[i]),dim=0).squeeze()
    self.weights = w
    return self.weights

  def get_S(self,Y):
    S = torch.empty((self.d_,self.branches))
    for i in range(self.branches):
      S[:,i] = Y[i].sum(dim=0).squeeze()
    return S

  def forward(self,l): #l = [X_list,AX_list,A2X_list,A3X_list]
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
  def __init__(self,d,r,num_classes):
    super(GraphSage, self).__init__()
    self.d = d
    self.r = r
    self.num_classes = num_classes

    self.layers = [SAGEConv(d,d,'mean') for _ in range(self.r)]

    self.classify = nn.Sequential(nn.Linear(d,d),nn.LeakyReLU(),nn.Linear(d,num_classes))
    
  def forward(self,X_list,graph_list):
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
  def __init__(self,d,r,num_classes):
    super(GIN, self).__init__()
    self.d = d
    self.r = r
    self.num_classes = num_classes
    self.MLP_layers = [nn.Sequential(nn.Linear(self.d,self.d),nn.LeakyReLU(),nn.Linear(self.d,self.d),nn.LeakyReLU()) for _ in range(r)]

    self.GIN_layers = [GINConv(self.MLP_layers[i],'sum',init_eps=0.1) for i in range(r)]

    self.classify = nn.Sequential(nn.Linear(r*d,d),nn.LeakyReLU(),nn.Linear(d,num_classes))
    
  def forward(self,X_list,graph_list):
    C = torch.empty((len(X_list),self.num_classes))
    for i in range(len(X_list)):
      Y = []
      y = X_list[i]
      sum = torch.empty((self.d,self.r))
      for j in range(self.r):
        y = L_Relu(self.GIN_layers[j](graph_list[i],y))
        Y.append(y)
        sum[:,j] = y.sum(dim=0)

      em = sum.reshape(1,-1)

      cl = self.classify(em)
      
      C[i] = cl
    return C
