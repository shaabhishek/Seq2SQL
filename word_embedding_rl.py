import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# This uses fixed glove embeddings.
# download glove 300D from Stanfordcore NLP

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, SQL_TOK,
            our_model):
        super(WordEmbedding, self).__init__()
        self.N_word = N_word
        self.our_model = our_model
        self.SQL_TOK = SQL_TOK
        self.word_emb = word_emb
        print("Using fixed embedding")


    def gen_x_batch(self, q, col):
        """
        :param question tokens = the question is tokenized as a list
                    E.g: ['what', 'station', 'aired', 'a', 'game', 'at', '9:00', 'pm', '?']
        
        :param col: Column header tokens = each head is a list of words it
                    contains (small letters)
                    E.g: ['Time', 'Big Ten Team'] => [['time'], ['big', 'ten', 'team']]
        
        :return val_inp_var: (Variable) a torch.autograd Variable of shape (batch_size, max_token_num, embedding_vector_len)
                            each value corresponds to a embedded value of a token for the specific training example
        
        :return val_len: (array) array of shape=(batch_size,) containing number of tokens for each training example
        """
        B = len(q) #Batch size
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            q_val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_q] #return the embedding for each word in question, and zero vector if it doesn't exist in embedding dict
            if self.our_model:
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                one_col_all = [x for toks in one_col for x in toks+[',']] #convert one_col from 2d list to 1d list. Each column is delimited by a ','
                col_val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_col_all] #return the embedding for each word in flattened column tokens list, and zero vector if it doesn't exist in embedding dict
                
                # each entry of val_embs is embedded version of SQL_TOK + col_tokens + (emptytoken) + q_tokens + (emptytoken)
                val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + 
                                    col_val +
                                    [np.zeros(self.N_word, dtype=np.float32)] + 
                                    q_val + 
                                    [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1 #length of val_embs for this question
        
        max_len = max(val_len) #longest training example (in terms of tokens)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32) #shape= (batch_size, max_token_num, embedding_vector_len)
        for i in range(B): #loop for each question
            for t in range(len(val_embs[i])): #loop for every token in the question
                val_emb_array[i,t,:] = val_embs[i][t] #move the content from the list to the array

        # convert the numpy array into a torch tensor.
        # both the objects point to the same memory location (changing one will change the other)
        val_inp = torch.from_numpy(val_emb_array) 
            
        val_inp_var = Variable(val_inp) #Wrap the tensor into a Variable. Required for auto-gradient computation
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        """"
        :return name_len: (np.array) number of tokens for each column header of the batch. shape=[size_of_column_header_space_of_cols,]

        :return name_inp_var: (Variable) shape = [size_of_column_header_space_of_cols, max_tokens_in_a_column_name, embedding_vector_len]
                            Entry (x,y,:) is the embedding values of xth header of batch's yth token. Most headers have 1 token but some have more.
        """
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols #flatten cols into a single list containing all the col_names in the batch
            col_len[b] = len(one_cols) #get number of column headers for this example

        # Get a Variable of shape=(len(names), max_tokens_in_a_column_name, embedding_vector_len)
        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list) #total number of (column) headers in the lists

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list): # for each in column name in batch
            
            # Get embedding value for each token in column name (one_str)
            val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val) #embedded values of tokens 
            val_len[i] = len(val) #how many tokens in the column name?
        max_len = max(val_len) #max tokens in a column name

        # This piece of code is similar to the one in function gen_x_batch
        # See comments there to get an idea of what is happening
        val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i,t,:] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
