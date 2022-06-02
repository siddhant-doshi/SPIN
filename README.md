# Simple and Parallel Graph Isomorphism Network 

We propose a generic GNN model that performs parallel neighborhood aggregations (referred to as PA-GNNs) through a bank of graph operators arranged in 
parallel, named SPIN, that pools node-level representations from each branch through branch-level readouts, and finally 
pools graph embeddings from multiple branches through a graph-level readout function. We present two variants of SPIN with and
without an attention mechanism at the branch-level readouts to capture the most relevant node features at each branch. 

Here, we detail about the implementation of SPIN and the ways to reproduce the results.

### Summary of the files: 

- Brain datasets: Folder that contains the raw input files of the brain datasets that we validate SPIN on. Provide these .nel files at the respective location if one wishes to work with these datasets.
- dataset_splits: Folder that contains the pre-defined datasplits for each fold (as we perform a 10-fold cross validation). The support to load these .json files is already there in the code. Provide the correct path for these files and set the flag "loadDataSplits = True" to use them. For the brain datasets, we generate the folds using random seed 17.
- datasets.py - It is the python script that performs the data preprocessing, which involves generating the features, the neighborhood information, and creating batches.
- models.py - The actual implementation of SPIN, and all the other existing GNN models that we use for comparing SPIN with are implemented in this script. Accordingly, if anyone wishes to do some manual changes and play around with the models, should be done in this file. 
- train.py - The training (with early stopping using a patience factor) and the evaluation function is implemented here. 
- main.py - The main file that needs to be run with all the other supporting files in place while importing. 
### The steps involved to reproduce the results are:
1. We would need to initialize the datasetName from the options provided 
#choices_set1|chemical| = ["ENZYMES","NCI1","PROTEINS_full","DD","ogbg-molhiv"]
#set2|social| = ["IMDB-BINARY","IMDB-MULTI","COLLAB","REDDIT-BINARY","REDDIT-MULTI-5K"]
#set3|brain| = ["OHSU","Peking_1"]. Advice to use the exact name string.
2. If using pre-defined datasplits, set the flag loadDataSplits appropriately and give the path of the split .json file in the variable predefinedSplitsFilepath.
3. Decide the foldIndex in [0,9] as we intend to perform a 10-fold cross validation.
4. Select the rHop value and the batchSize to start the precomputations and generate all the necessary information.
5. Set the hyperparas in hyperpara_dict.
6. Train the model.

The preprint of this complete work is available at https://arxiv.org/abs/2111.11482.
