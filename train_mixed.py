import json
import torch
from utils_rl import *
from seq2sql_rl import Seq2SQL
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# This is the training model - where there is no loading of model state.
# Training on CE losses first followed by RL
# best model is saved for test.
# saved model attached along with the code
# is generated from this model.
N_word=300
B_word=42
BATCH_SIZE=64
loss_list = []
epoch_num = []
TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
reinforce = True


sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset()

word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word))
model = Seq2SQL(word_emb, N_word=N_word)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay = 0)
#Adam default lr = 0.001, Can also try 0.01/0.05
agg_m, sel_m, cond_m = best_model_name()


init_acc = epoch_acc(model, BATCH_SIZE,
                    val_sql_data, val_table_data, TRAIN_ENTRY)
best_agg_acc = init_acc[1][0]
best_agg_idx = 0
best_sel_acc = init_acc[1][1]
best_sel_idx = 0
best_cond_acc = init_acc[1][2]
best_cond_idx = 0
print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s'%\
                    init_acc)
    
for i in range(400):
    print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
    epoch_num.append(i+1)
    b = epoch_train(model, optimizer, BATCH_SIZE, 
                        sql_data, table_data, TRAIN_ENTRY)
    print(' Loss = %s'%b)
    loss_list.append(b)
    print(' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
                        model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
    val_acc = epoch_acc(model,
                        BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
    print(' Dev acc_qm: %s\n   breakdown result: %s'%val_acc)
    if val_acc[1][0] > best_agg_acc:
        best_agg_acc = val_acc[1][0]
        best_agg_idx = i+1
        torch.save(model.agg_pred.state_dict(), agg_m)    
                
    if val_acc[1][1] > best_sel_acc:
        best_sel_acc = val_acc[1][1]
        best_sel_idx = i+1
        torch.save(model.sel_pred.state_dict(), sel_m)    
                
    if val_acc[1][2] > best_cond_acc:
        best_cond_acc = val_acc[1][2]
        best_cond_idx = i+1
        torch.save(model.cond_pred.state_dict(), cond_m)    
                        
    print(' Best val acc = %s, on epoch %s individually'%(
                        (best_agg_acc, best_sel_acc, best_cond_acc),
                        (best_agg_idx, best_sel_idx, best_cond_idx)))

bestince = (best_agg_acc, best_sel_acc, best_cond_acc)
#plt.plot(epoch_num,loss_list, 'ro')
#plt.xlabel('Epoch_Num')
#plt.ylabel('Loss')
#plt.show()
#plt.savefig('lossdropmixed_curve.png')

best_acc = 0.0
best_idx = -1
print("Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s"% \
                epoch_acc(model, BATCH_SIZE, val_sql_data,\
                val_table_data, TRAIN_ENTRY))
print("Init dev acc_ex: %s"%epoch_exec_acc(
                model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
reinforce = True   
for i in range(500):

    print('Epoch in RL train %d @ %s'%(i+1, datetime.datetime.now()))
       
    print(' Avg reward = %s'%epoch_reinforce_train(
                model, optimizer, BATCH_SIZE, sql_data, table_data, TRAIN_DB))
    print(' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
                        model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
    print(' dev acc_qm: %s\n   breakdown result: %s'% epoch_acc(
                model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY))
    exec_acc = epoch_exec_acc(
                    model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
    print(' dev acc_ex: %s'%exec_acc)
    if exec_acc > best_acc:
        best_acc = exec_acc
        best_idx = i+1
        torch.save(model.agg_pred.state_dict(), agg_m)
        torch.save(model.sel_pred.state_dict(), sel_m)
        torch.save(model.cond_pred.state_dict(), cond_m)
    print(' Best exec acc = %s, on epoch %s'%(best_acc, best_idx))
