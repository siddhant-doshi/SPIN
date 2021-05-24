pip install dgl #-cu101


import matplotlib.pyplot as plt
import seaborn as sns

from datasets import *
from models import *
from train import *

#choices_set1 = ["ENZYMES","NCI1","PROTEINS_full","DD"]
#set2 = ["IMDB-BINARY","IMDB-MULTI","COLLAB","REDDIT-BINARY","REDDIT-MULTI-5K"]
#copy the exact dataset name
datasetName = "IMDB-MULTI"

#if you want to load the predefined datasplits: 1) set the flag to True and 2) give the filepath
#The dataset splits used in our experiements can be found at: https://github.com/diningphil/gnn-comparison/tree/master/data_splits
loadDataSplits = True
predefinedSplitsFilepath = '/content/imdb_multi.json'
datasetObj = load_dataset(datasetName,load_datasplits = loadDataSplits,predefined_splits_filepath = predefinedSplitsFilepath)
#this is the base dataset obj

#select the fold and create batches
#selected fold is considered as the test set, and remaining as the complete training set
#training set is internally divided in validation and training 
#if predefined splits are use (e.g., from the link we gave above), they will be automatically considered
foldIndex = 1
dataFold = split_data(datasetObj,foldIndex,val_split = 0.1,validation_split_seed=19) #object containing the train, validation and test graphs with corresponding labels
#dataObj.split_data(fold_index_k=foldIndex,val_split = 0.1,validation_split_seed=19)

rHop = 2 #hyperpara
batchSize = 32
dataFold.create_batches(datasetObj,r_hop = rHop,batch_size = batchSize,save_batches = False,path = '')
#dataFold.train_batches = [X_batches, AX_batches, ... , ArX_batches, label_batches, graph_batches]
#dataFold.train_batches[i][j][k] : i corresponds to X_batches (or AX_batches and so), j: batch num, k: Data index in the jth batch

intermediate_dim = 32
d_hat = 16
net = SPIN(datasetObj.input_feat_dim,d_hat,datasetObj.num_classes,rHop,intermediate_dim,agg='concat',attention=False)

#train model
num_epochs = 500
patience_factor = 300 #if the validation does not achieve a new peak in <patience_factor> epochs, we stop.
results,train_specs = train_model(dataFold,net,patience_factor,num_epochs,loss = "Cross_entropy",optimizer = "Adam",learning_rate = 5e-3)

print (results,end='\n',sep='\n')

sns.set_theme()
l = len(train_specs['epochwise_loss'])
plt.plot(range(l),train_specs['epochwise_train_acc'],color='green')
plt.plot(range(l),train_specs['epochwise_val_acc'],color='royalblue')
plt.plot(range(l),results['test_accuracy']*np.ones(l),color='crimson',linestyle = 'dashdot')