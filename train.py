
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(logits,labels):
  pred_train = [torch.argmax(logits[j]).item() for j in range(len(logits))]
  return accuracy_score(labels,pred_train)

def train_model(dataFold,net,patience_factor,num_epochs,loss = "Cross_entropy",optimizer = "Adam",L2 = 0,learning_rate = 1e-3):
  avg_epochwise_loss,epochwise_val_loss,avg_epochwise_train_acc,avg_epochwise_val_acc,avg_epochwise_test_acc = [],[],[],[],[]
  best_model = {"Epoch_index":0,"model":net,"train_accuracy":0.0,"val_accuracy":0.0,"test_accuracy":0.0}
  if (loss == "Cross_entropy") : CE_loss = nn.CrossEntropyLoss()
  if (optimizer == "Adam"):
    if (L2):
      optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay = L2)
    else:
      optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  epoch_num = 0
  p = 0
  try:
    while (epoch_num <= num_epochs and p < patience_factor):
      l,t_acc=0,0
      net.train()
      for i in range(dataFold.num_batches):
        logits = net([dataFold.train_batches[j][i] for j in range(len(dataFold.train_batches[:-2]))])
        loss = CE_loss(logits,dataFold.train_batches[-2][i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b_acc = evaluate(logits,dataFold.train_batches[-2][i])
        print ("epoch = %d | batch = %d | loss = %f | batch_accuracy = %f "%(epoch_num,i,loss.item(),b_acc))
        l+=loss.item()
        t_acc+=b_acc
      
      net.eval()
      logits_val = net([dataFold.val_batches[j][0] for j in range(len(dataFold.val_batches[:-2]))])
      val_loss = CE_loss(logits_val,dataFold.val_batches[-2][0])
      acc_val = evaluate(logits_val,dataFold.val_batches[-2][0])

      logits_test = net([dataFold.test_batches[j][0] for j in range(len(dataFold.test_batches[:-2]))])
      acc_test = evaluate(logits_test,dataFold.test_batches[-2][0])

      avg_epochwise_loss.append(l/dataFold.num_batches)
      epochwise_val_loss.append(val_loss)
      avg_epochwise_train_acc.append(t_acc/dataFold.num_batches)
      avg_epochwise_val_acc.append(acc_val)
      print ("-----------epoch = %d | avg_loss = %f | avg_epoch_accuracy = %f | val_acc = %f"%(epoch_num,l/dataFold.num_batches,t_acc/dataFold.num_batches,acc_val))

      if (best_model["val_accuracy"] < acc_val):
        best_model["Epoch_index"] = epoch_num
        best_model["model"] = net
        best_model["train_accuracy"] = t_acc/dataFold.num_batches
        best_model["val_accuracy"] = acc_val
        best_model["test_accuracy"] = acc_test
        p = 0
      else:
        p += 1
      epoch_num += 1
      training_specs = {"epochwise_loss":avg_epochwise_loss,"epochwise_train_acc":avg_epochwise_train_acc,"epochwise_val_acc":avg_epochwise_val_acc}
    
    if (p >= patience_factor) : print ("Patience factor termination")
    return best_model,training_specs
  except KeyboardInterrupt:
    return best_model,training_specs