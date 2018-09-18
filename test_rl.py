import json
import torch
from utils_rl import *
from seq2sql_rl import Seq2SQL
import numpy as np
import datetime


N_word=300
B_word=42
BATCH_SIZE=64
TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset()

word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word)) 
model = Seq2SQL(word_emb, N_word=N_word)

    
agg_m, sel_m, cond_m = best_model_name()
print("Loading from %s"%agg_m)
model.agg_pred.load_state_dict(torch.load(agg_m))
print("Loading from %s"%sel_m)
model.sel_pred.load_state_dict(torch.load(sel_m))
print("Loading from %s"%cond_m)
model.cond_pred.load_state_dict(torch.load(cond_m))

print("Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY))
print("Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY))
print("Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB))
