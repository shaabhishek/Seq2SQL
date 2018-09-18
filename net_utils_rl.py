import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# These are created to pack the sequences and save the hidden states of the LSTM.
# In order to reduce repeated lines of code through out the model.
# It is defined once and used for all three LSTMs that are used in the model.
# All it does is take the batch of sequences and pad them before 
# sending them through lstm and collect the hidden states
# once encoded.

def run_lstm(lstm, inp, inp_len, hidden=None):
    """
    :param lstm: LSTM object
    
    :param inp: (Variable) embedded versions of tokens as a batch (length_of_one_token = dim 2).
                shape=[batch_size, max_tokens_per_example_in_batch, length_of_one_token]
    
    :param inp_len: (array) number of tokens for each training example in batch
    
    :param hidden: (tuple) hidden_state, cell_state to input to LSTM

    :return ret_s: (Variable) output of each cell of LSTM. shape = shape of inp with last dim = N_h
    
    :return ret_h: (tuple) (hidden_state, cell_state) of the last cell
    """
    # sort_perm = np.array(sorted(list(range(len(inp_len))),
    #     key=lambda k:inp_len[k], reverse=True)) #Get indices of sorted inp_len in decreasing order. Slow.., replaced by code below
    
    # Get indices of sorted inp_len in decreasing order.
    # copy is needed, otherwise indexing of hidden doesn't work while decoding
    sort_perm = np.argsort(inp_len)[::-1].copy() 
    sort_inp_len = inp_len[sort_perm] #inp_len in decreasing order
    sort_perm_inv = np.argsort(sort_perm) #to return 'inp' to same order in which it was received
    
    # Get a PackedSequence object which contains the non-zero entries
    # of 'inp' packed in a single variable with length of each batch entry stored as metadata
    # takes input in decreasing order of length
    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True) 
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden) #forwardprop on the lstm. sort_ret_h is tuple(hidden_state, cell_state)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv] #Unpack output of LSTM to padded values with each batch entry of fixed length
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the column names.
    # name_hidden is confusing. It is not the hidden state but the output of the encoder
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len) #shape(name_hidden) = [total_tokens_in_batch, max_tokens_in_a_column, N_h=100]
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1] #shape(name_out) = [total_tokens_in_batch, N_h=100]. Get last token's value for each column.
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_() #generate zero tensor of shape
    
    st = 0
    for idx, cur_len in enumerate(col_len): #for each batch entry (training example), cur_len is number of columns
        ret[idx, :cur_len] = name_out.data[st:st+cur_len] #fill in the last token values for all columns in the example
        st += cur_len
    ret_var = Variable(ret) #[batch_size, max_number_of_columns_in_an_example, N_h]

    return ret_var, col_len
