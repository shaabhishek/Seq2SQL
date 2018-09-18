import json
import torch
from utils_rl_200h import *
from seq2sql_rl_200h import Seq2SQL
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# similar to the other two models. Here only the
# model state is saved.

N_word=300
B_word=42
BATCH_SIZE=64
loss_list = []
epoch_num = []
TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
reinforce = True
if reinforce:
    learning_rate = 0.01 # or 0.05 or 0.001
else:
    learning_rate = 0.001

sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset()

word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word))
model = Seq2SQL(word_emb, N_word=N_word)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 0)
def count_parameters(model):
    
    return [sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(n.numel() for n in model.parameters() if not n.requires_grad)]
def checkgrad(model):
    
    return [p.grad() for p in model.parameters() if p.requires_grad]

agg_m, sel_m, cond_m = best_model_name()


if reinforce:
    agg_lm, sel_lm, cond_lm = best_model_name(for_load = True)
    print("Loading from %s"%agg_m)
    model.agg_pred.load_state_dict(torch.load(agg_m))
    print("Loading from %s"%sel_m)
    model.sel_pred.load_state_dict(torch.load(sel_m))
    print("Loading from %s"%cond_m)
    model.cond_pred.load_state_dict(torch.load(cond_m))

    best_acc = 0.0
    best_idx = -1
    print("Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s"% \
                epoch_acc(model, BATCH_SIZE, val_sql_data,\
                val_table_data, TRAIN_ENTRY))
    print("Init dev acc_ex: %s"%epoch_exec_acc(
                model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
    
    torch.save(model.cond_pred.state_dict(), cond_lm)
    torch.save(model.agg_pred.state_dict(), agg_lm)
    torch.save(model.sel_pred.state_dict(), sel_lm)
    for i in range(400):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
       
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
            
            torch.save(model.cond_pred.state_dict(), cond_lm)
            torch.save(model.agg_pred.state_dict(), agg_lm)
            torch.save(model.sel_pred.state_dict(), sel_lm)
        print(' Best exec acc = %s, on epoch %s'%(best_acc, best_idx))

else:
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
    #if TRAIN_AGG:
    torch.save(model.agg_pred.state_dict(), agg_m)
                
    #if TRAIN_SEL:
    torch.save(model.sel_pred.state_dict(), sel_m)
                
    #if TRAIN_COND:
    torch.save(model.cond_pred.state_dict(), cond_m)
                
    for i in range(100):
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

#plt.plot(epoch_num,loss_list, 'ro')
#plt.xlabel('Epoch_Num')
#plt.ylabel('Loss')
#plt.show()
#plt.savefig('lossdrop200h_curve.png')