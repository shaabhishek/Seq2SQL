import json
import torch
from utils_rl import *
from seq2sql_rl import Seq2SQL
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# This model is similar to other two train models. Only difference here is,
# along with model state, even optimizer state
# is saved along the way.
# Test model has only model state loaded. Change
# test code accordingly if intend to use this train code.

N_word=300
B_word=42
BATCH_SIZE=15
loss_list = []
epoch_num = []
TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
reinforce = True

if reinforce:
    learning_rate = 1e-4
else:
    learning_rate = 1e-3

# sql_data, table_data, val_sql_data, val_table_data, \
#             test_sql_data, test_table_data, \
#             TRAIN_DB, DEV_DB, TEST_DB = load_dataset()

sqldataset = SQLDataset('data_resplit/train.jsonl', 'data_resplit/tables.jsonl')
sql_dataloader = DataLoader(sqldataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)

word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word))
model = Seq2SQL(word_emb, N_word=N_word)

torch.set_num_threads(4)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 0)

agg_m, sel_m, cond_m = best_model_name()
def count_parameters(model):
    
    return [sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(n.numel() for n in model.parameters() if not n.requires_grad)]
if reinforce:
    agg_lm, sel_lm, cond_lm = best_model_name(for_load = True)
    checkpoint_agg = torch.load(agg_m)
    checkpoint_sel = torch.load(sel_m)
    checkpoint_cond = torch.load(cond_m)
    
    print("Loading from %s"%agg_m)
    print("Loading from %s"%sel_m)
    print("Loading from %s"%cond_m)
    model.agg_pred.load_state_dict(checkpoint_agg)
    model.sel_pred.load_state_dict(checkpoint_sel)
    model.cond_pred.load_state_dict(checkpoint_cond)
    # optimizer.load_state_dict(checkpoint_cond['optimizer'])
    
    #para = count_parameters(model)
    #print(para)
    
    #model.agg_pred.load_state_dict(torch.load(agg_m))
    #model.sel_pred.load_state_dict(torch.load(sel_m))
    #model.cond_pred.load_state_dict(torch.load(cond_m))
    
    best_acc = 0.0
    best_idx = -1
    # print("Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s"% \
    #             epoch_acc(model, BATCH_SIZE, val_sql_data,\
    #             val_table_data, TRAIN_ENTRY))
    # print("Init dev acc_ex: %s"%epoch_exec_acc(
    #             model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
    

    stateagg = model.agg_pred.state_dict()
    statesel = model.sel_pred.state_dict()
    statecond = model.cond_pred.state_dict()
    torch.save(stateagg, agg_lm)
    torch.save(statesel, sel_lm)
    torch.save(statecond, cond_lm)

    # stateagg = {'state_dict': model.agg_pred.state_dict(),
    #          'optimizer': optimizer.state_dict()}
    # statesel = {'state_dict': model.sel_pred.state_dict(),
    #          'optimizer': optimizer.state_dict() }
    # statecond = {'state_dict': model.cond_pred.state_dict(),
    #          'optimizer': optimizer.state_dict() }
    
    #torch.save(model.cond_pred.state_dict(), cond_lm)
    #torch.save(statecond, cond_lm)
    for i in range(500):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        # if i==49:
        #     import pdb; pdb.set_trace()
        # avg_reward = epoch_reinforce_train(model, optimizer, BATCH_SIZE, sql_data, table_data, TRAIN_DB)
        avg_reward = epoch_reinforce_train(model, optimizer, BATCH_SIZE, sql_dataloader)
        print('Avg reward = %s'%avg_reward)
        
        # accuracy_train_total, accuracy_train_split = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY)
        # print(' Train accuracy_qm: %s\n   breakdown result: %s'%(accuracy_train_total, accuracy_train_split))
        
        # accuracy_dev_total, accuracy_dev_split = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        # print(' Dev accuracy_qm: %s\n   breakdown result: %s'% (accuracy_dev_total, accuracy_dev_split))
        
        # exec_acc = epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)

        # print(' Dev acc_execution: %s'%exec_acc)
        
        # if exec_acc > best_acc:
        #     best_acc = exec_acc
        #     best_idx = i+1
            
        #     stateagg = model.agg_pred.state_dict()
        #     statesel = model.sel_pred.state_dict()
        #     statecond = model.cond_pred.state_dict()
        #     torch.save(statecond, cond_lm)
        #     torch.save(statesel, sel_lm)
        #     torch.save(stateagg, agg_lm)
        # print(' Best exec acc = %s, on epoch %s'%(best_acc, best_idx))

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
    
    stateagg = model.agg_pred.state_dict()
    statesel = model.sel_pred.state_dict()
    statecond = model.cond_pred.state_dict()
    torch.save(stateagg, agg_m)
    torch.save(statesel, sel_m)
    torch.save(statecond, cond_m)  

    # stateagg = {'state_dict': model.agg_pred.state_dict(),
    #          'optimizer': optimizer.state_dict()}
    # statesel = {'state_dict': model.sel_pred.state_dict(),
    #          'optimizer': optimizer.state_dict() }
    # statecond = {'state_dict': model.cond_pred.state_dict(),
    #          'optimizer': optimizer.state_dict() }
               
    for i in range(2):
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
        print(' Val acc_qm: %s\n   breakdown result: %s'%val_acc)
        if val_acc[1][0] > best_agg_acc:
            best_agg_acc = val_acc[1][0]
            best_agg_idx = i+1
            
            torch.save(stateagg, agg_m)
                        
                
        if val_acc[1][1] > best_sel_acc:
            best_sel_acc = val_acc[1][1]
            best_sel_idx = i+1
           
            torch.save(statesel, sel_m)
                        
                
        if val_acc[1][2] > best_cond_acc:
            best_cond_acc = val_acc[1][2]
            best_cond_idx = i+1
            
            torch.save(statecond, cond_m)
                        
        print(' Best val acc = %s, on epoch %s individually'%(
                        (best_agg_acc, best_sel_acc, best_cond_acc),
                        (best_agg_idx, best_sel_idx, best_cond_idx)))

#plt.plot(epoch_num,loss_list, 'ro')
#plt.xlabel('Epoch_Num')
#plt.ylabel('Loss')
#plt.show()
#plt.savefig('lossdrop_curve.png')