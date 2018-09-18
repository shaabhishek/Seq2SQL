import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils_rl import run_lstm, col_name_encode
# This piece of code is for prediction of selected column.
# Only one column name is the output to this LSTM.

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num):
        super(SelPredictor, self).__init__()
        self.max_tok_num = max_tok_num # Taken as 200
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        #Many one LSTM - output is prob distribution over the column names.
        self.sel_att = nn.Linear(N_h, 1) #bias need not be present. paper doesn't use any
        # one column should be predicted for the selection column part of the query
        #encode all the column tokens seperately
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        # self.softmax = nn.Softmax()


    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        """
        :param x_emb_var: array of tokens for the LSTM to choose the selector from.
                         shape=(batch_size, max_tokens_in_a_training_example, embedding_vector_len)
        :param x_len: array of number of tokens in each training example
        :param col_inp_var: embedding of each token in each column.
                         shape=(col_name_len, max_tokens_in_a_column_name, embedding_vector_len)
        :param col_name_len array of number of tokens in each column name (across the whole batch)
        :param col_len array of number of column names for each example
        :param col_num array of number of column names for each example
        """
        B = len(x_emb_var) #batch size
        max_x_len = max(x_len)

        # run the encoder. pass the LSTM object (which acts as encoder) to the function too.
        # Relevant reference in paper: In order to produce the representations for the columns, we first encode each column name with a LSTM
        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)

        # The input to the encoder are the embeddings corresponding to words in the input sequence.
        # We denote the output of the encoder by h_enc, where h_enc[t]
        # is the state of the encoder corresponding to the t-th word in the input sequence.
        h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len) #shape=[batch_size, max_tokens_in_a_training_example, N_h]

        # scalar attention exactly as mentioned in the paper.
        att_val = self.sel_att(h_enc).squeeze() #sel_attn reduces last dimension (size L_h) to size 1. squeeze removes that dimension and converts tensor to 2D
        for idx, num in enumerate(x_len): #for each training example. num is number of tokens in the training example
            if num < max_x_len:
                att_val[idx, num:] = -100 #manually set attention score to -100 for entries where tokens are not present
        att = F.softmax(att_val, 1) #normalize to sum to 1 for each example
        K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1) #add dimension 2 to att, then expand dimension 2 to size 100, then multiply with h_enc element wise and sum of dimension 1
        K_sel_expand=K_sel.unsqueeze(1) #insert new dimension 1


        sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col) ).squeeze() #apply linear layers, sum and then apply sel_out layer. shape(sel_score) = [batch_size, max_number_of_columns_in_an_example]
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100 #manually set attention score to -100 for entries where columns are not present

        return sel_score
